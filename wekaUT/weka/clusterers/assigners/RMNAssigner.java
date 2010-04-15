/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    RMNAssigner.java
 *    RMN assignment for K-Means
 *    Copyright (C) 2004 Misha Bilenko, Sugato Basu
 *
 */

package weka.clusterers.assigners; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.core.metrics.*;
import  weka.clusterers.*;
import  weka.clusterers.assigners.*;
import  rmn.*;


public class RMNAssigner extends MPCKMeansAssigner {
  /** Inference can be single-pass approximate or multi-pass approximate */
  protected boolean m_singlePass = true; 
  /** scaling factor for exponent */
  protected double m_expScalingFactor = 10;
  /** scaling factor for constraint weights */
  protected double m_constraintWeight = 1000;

  public RMNAssigner() {
    super();
  }

  public RMNAssigner(MPCKMeans clusterer) {
    super(clusterer);
  }

  /** This is a sequential assignment method */
  public boolean isSequential() {
    return false;
  }

  /** small value to replace 0 in some places, to avoid numerical
      underflow */
  public double m_epsilon;

  /** The main method
   *  @return the number of points that changed assignment
   */
  public int assign() throws Exception {    
    int moved = 0;
    boolean verbose = m_clusterer.getVerbose();
    m_epsilon = 1e-9;

    Metric metric = m_clusterer.getMetric();
    LearnableMetric[] metrics = m_clusterer.getMetrics();
    boolean useMultipleMetrics = m_clusterer.getUseMultipleMetrics();
    Instances instances = m_clusterer.getInstances();
    Instances centroids = m_clusterer.getClusterCentroids();
    int numInstances = instances.numInstances();
    int numClusters = m_clusterer.getNumClusters();

    Random random = new Random(m_clusterer.getRandomSeed()); // initialize random number generator

    // create factor graph
    FactorGraph fg = new FactorGraph();

    // create variable nodes
    Variable[] vars = new Variable[numInstances];
    for (int i=0; i<numInstances; i++) {
      String name = "var:" + i;
      vars[i] = new Variable(name, numClusters);
    }

    // create centroid potential nodes
    double maxSim = 0;
    double[][] simMatrix = new double[numInstances][numClusters];

    for (int i=0; i<numInstances; i++) {
      Instance instance = instances.instance(i);
      // fill up potential table using current distances
      for (int centroidIdx=0; centroidIdx<numClusters; centroidIdx++) {
	Instance centroid = centroids.instance(centroidIdx);
	if (!m_clusterer.isObjFunDecreasing()) { // increasing obj. function
	  if (useMultipleMetrics) { // multiple metrics
	    simMatrix[i][centroidIdx] = metrics[centroidIdx].similarity(instance, centroid);
	  } else {
	    simMatrix[i][centroidIdx] = metric.similarity(instance, centroid);
	  }
	} else { // decreasing obj. function
	  if (useMultipleMetrics) { // multiple metrics
	    simMatrix[i][centroidIdx] = metrics[centroidIdx].distance(instance, centroid);
	  } else {
	    simMatrix[i][centroidIdx] = metric.distance(instance, centroid);
	  }
	  if (metric instanceof WeightedEuclidean || metric instanceof WeightedMahalanobis) {
	    simMatrix[i][centroidIdx] *= simMatrix[i][centroidIdx];
	  }
	}
	if (maxSim < simMatrix[i][centroidIdx]) {
	  maxSim = simMatrix[i][centroidIdx];
	}
	if (verbose) {
	  System.out.println("simMatrix[" + i + "," + centroidIdx + "]: " + simMatrix[i][centroidIdx] + ", MaxSim: " + maxSim);
	}
      }
    }

    m_expScalingFactor = maxSim;

    for (int i=0; i<numInstances; i++) {
      double[] weightVector = new double[numClusters];
      // fill up potential table using current distances
      for (int centroidIdx=0; centroidIdx<numClusters; centroidIdx++) {
	if (!m_clusterer.isObjFunDecreasing()) { // increasing obj. function
	  weightVector[centroidIdx] = Math.exp(simMatrix[i][centroidIdx]/m_expScalingFactor);
	  if (weightVector[centroidIdx] < m_epsilon) {
	    weightVector[centroidIdx] = m_epsilon;
	  }
	} else {
	  weightVector[centroidIdx] = Math.exp(-simMatrix[i][centroidIdx]/m_expScalingFactor);
	  if (weightVector[centroidIdx] < m_epsilon) {
	    weightVector[centroidIdx] = m_epsilon;
	  }
	}
	
	if (verbose) {
	  System.out.println("Centroid weight[" + centroidIdx + "] for instance: " + i + " = " + weightVector[centroidIdx]);
	}
      }

      // create centroid potential node
      PotentialFactory1 pf1 = new PotentialFactory1(weightVector);
      Potential pot = pf1.newInstance();
      
      // add edges between potential and variable nodes
      Variable[] node = new Variable[1];
      node[0] = vars[i];
      fg.addEdges(pot, node);
    }

    // create ML and CL potential nodes
    HashMap constraintsHash = m_clusterer.getConstraintsHash();
    if (constraintsHash != null) {
      System.out.println("Creating constraint potential nodes");
      Set pointPairs = (Set) constraintsHash.keySet(); 
      Iterator pairItr = pointPairs.iterator();
      
      // iterate over the pairs in ConstraintHash
      while( pairItr.hasNext() ){
	InstancePair pair = (InstancePair) pairItr.next();
	Instance instance1 = instances.instance(pair.first);
	Instance instance2 = instances.instance(pair.second);
	int linkType = ((Integer) constraintsHash.get(pair)).intValue();

	double cost = 0;
	if (linkType == InstancePair.MUST_LINK) {
	  cost = m_clusterer.getMustLinkWeight();
	} else if (linkType == InstancePair.CANNOT_LINK) {
	  cost = m_clusterer.getCannotLinkWeight();
	}

	if (verbose) {
	  System.out.println(pair + ": type = " + linkType);
	}

	double[][] weightMatrix = new double[numClusters][numClusters];
	if( linkType == InstancePair.MUST_LINK ){ // create ML potential node
	  for (int centroidIdx1=0; centroidIdx1<numClusters; centroidIdx1++) {
	    for (int centroidIdx2=0; centroidIdx2<numClusters; centroidIdx2++) {
	      // fill up potential table using current distances	      
	      if (centroidIdx1!=centroidIdx2) {
		double weight = 0;
		if (metric instanceof WeightedDotP) {
		  if (useMultipleMetrics) {  // split penalty in half between the two involved clusters
		    double sim1 = metrics[centroidIdx1].similarity(instance1, instance2);
		    weight -= 0.5 * cost * (1 - sim1);
		    double sim2 = metrics[centroidIdx2].similarity(instance1, instance2);
		    weight -= 0.5 * cost * (1 - sim2); 
		  } else {  // single metric for all clusters
		    double sim = metric.similarity(instance1, instance2);
		    weight -= cost * (1 - sim);
		  }
		  weightMatrix[centroidIdx1][centroidIdx2] = Math.exp(m_constraintWeight*m_expScalingFactor*weight);
		  weightMatrix[centroidIdx2][centroidIdx1] = Math.exp(m_constraintWeight*m_expScalingFactor*weight);
		} else if (metric instanceof KL) {
		  if (useMultipleMetrics) {  // split penalty in half between the two involved clusters
		    double penalty1 = ((KL) metrics[centroidIdx1]).distanceJS(instance1, instance2);
		    weight += 0.5 * cost * penalty1;
		    double penalty2 = ((KL) metrics[centroidIdx2]).distanceJS(instance1, instance2);
		    weight += 0.5 * cost * penalty2;
		  } else {  // single metric for all clusters
		    double penalty = ((KL) metric).distanceJS(instance1, instance2);
		    weight += cost * penalty;
		  }
		  weightMatrix[centroidIdx1][centroidIdx2] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		  weightMatrix[centroidIdx2][centroidIdx1] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		} else if (metric instanceof WeightedEuclidean || metric instanceof WeightedMahalanobis) {
		  if (useMultipleMetrics) {  // split penalty in half between the two involved clusters
		    double distance1 = metrics[centroidIdx1].distance(instance1, instance2);
		    weight += 0.5 * cost * distance1 * distance1;
		    double distance2 = metrics[centroidIdx2].distance(instance1, instance2);
		    weight += 0.5 * cost * distance2 * distance2;
		  } else {  // single metric for all clusters
		    double distance = metric.distance(instance1, instance2);
		    weight += cost * distance * distance;
		  }
		  weightMatrix[centroidIdx1][centroidIdx2] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		  weightMatrix[centroidIdx2][centroidIdx1] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		}
	      } else { // no constraint violation
		weightMatrix[centroidIdx1][centroidIdx2] = 1;
		weightMatrix[centroidIdx2][centroidIdx1] = 1;
	      }
	      
	      if (weightMatrix[centroidIdx1][centroidIdx2] < m_epsilon) {
		weightMatrix[centroidIdx1][centroidIdx2] = m_epsilon;
	      }
	      if (weightMatrix[centroidIdx2][centroidIdx1] < m_epsilon) {
		weightMatrix[centroidIdx2][centroidIdx1] = m_epsilon;
	      }
	      
	      if (verbose) {
		System.out.println("Link weight[" + centroidIdx1 + "," + centroidIdx2 + "] for pair: (" + pair.first + "," + pair.second + "," + linkType + ") = " + weightMatrix[centroidIdx1][centroidIdx2]);	      
	      }
	    }
	  }
	} else { // create CL potential node
	  for (int centroidIdx1 = 0; centroidIdx1 < numClusters; centroidIdx1++) {
	    for (int centroidIdx2 = 0; centroidIdx2 < numClusters; centroidIdx2++) {
	      // fill up potential table using current distances
	      if (centroidIdx1 == centroidIdx2) {
		double weight = 0;
		if (metric instanceof WeightedDotP) {
		  if (useMultipleMetrics) {  // centroidIdx1 == centroidIdx2
		    weight -= cost * metrics[centroidIdx1].similarity(instance1, instance2);
		  } else {  // single metric for all clusters
		    weight -= cost * metric.similarity(instance1, instance2);
		  }
		  weightMatrix[centroidIdx1][centroidIdx2] = Math.exp(m_constraintWeight*m_expScalingFactor*weight);
		  weightMatrix[centroidIdx2][centroidIdx1] = Math.exp(m_constraintWeight*m_expScalingFactor*weight);
		} else if (metric instanceof KL) {
		  if (useMultipleMetrics) {  // centroidIdx1 == centroidIdx2		    
		    double penalty = 2.0 - ((KL) metrics[centroidIdx1]).distanceJS(instance1, instance2);
		    weight += cost * penalty; 
		  } else {  // single metric for all clusters
		    double penalty = 2.0 - ((KL) metric).distanceJS(instance1, instance2);
		    weight += cost * penalty;
		  }
		  weightMatrix[centroidIdx1][centroidIdx2] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		  weightMatrix[centroidIdx2][centroidIdx1] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		} else if (metric instanceof WeightedEuclidean || metric instanceof WeightedMahalanobis) {
		  if (useMultipleMetrics) {  // centroidIdx1 == centroidIdx2
		    double maxDistance = metrics[centroidIdx1].distance(m_clusterer.m_maxCLPoints[centroidIdx1][0],
									m_clusterer.m_maxCLPoints[centroidIdx1][1]);
		    double distance = metrics[centroidIdx1].distance(instance1, instance2);
		    weight += cost * (maxDistance * maxDistance - distance * distance); 
		  } else {  // single metric for all clusters
		    double maxDistance =  metric.distance(m_clusterer.m_maxCLPoints[0][0],
							  m_clusterer.m_maxCLPoints[0][1]);
		    double distance = metric.distance(instance1, instance2);
		    weight += cost * (maxDistance * maxDistance - distance * distance); 
		  }
		  weightMatrix[centroidIdx1][centroidIdx2] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		  weightMatrix[centroidIdx2][centroidIdx1] = Math.exp(-m_constraintWeight*weight/m_expScalingFactor);
		}
	      } else { // no constraint violation
		weightMatrix[centroidIdx1][centroidIdx2] = 1;
		weightMatrix[centroidIdx2][centroidIdx1] = 1;
	      }

	      if (weightMatrix[centroidIdx1][centroidIdx2] < m_epsilon) {
		weightMatrix[centroidIdx1][centroidIdx2] = m_epsilon;
	      }
	      if (weightMatrix[centroidIdx2][centroidIdx1] < m_epsilon) {
		weightMatrix[centroidIdx2][centroidIdx1] = m_epsilon;
	      }

	      if (verbose) {
		System.out.println("Link weight[" + centroidIdx1 + "," + centroidIdx2 + "] for pair: (" + pair.first + "," + pair.second + "," + linkType + ") = " + weightMatrix[centroidIdx1][centroidIdx2]);
	      }
	    }
	  }
	}

	PotentialFactory2 pf2 = new PotentialFactory2(weightMatrix);
	Potential pot = pf2.newInstance();
	Variable[] nodePair = new Variable[2];
	// add edges between potential and variable nodes
	nodePair[0] = vars[pair.first];
	nodePair[1] = vars[pair.second];
	fg.addEdges(pot, nodePair);
      }
    }
    
    fg.allocateMessages();

    System.out.println("Doing MPE inference");
    if (m_singlePass) {
      fg.setMPE();  // Razvan's fast approximate computation 
    } else {
      fg.setExactMPE(); // Kevin Murphy's exact computation
    }

    /****/
    /**** Compare to default assigner */
    /****/

    // compare to default assigner
    SimpleAssigner simple = new SimpleAssigner(m_clusterer);
    int [] clusterAssignments = m_clusterer.getClusterAssignments();
    int [] oldAssignments = new int[numInstances];
    int [] simpleAssignments = new int[numInstances];

    // backup assignments before E-step
    for (int i=0; i<numInstances; i++) {
      oldAssignments[i] = clusterAssignments[i];
    }

    // get assignments with default E-step
    simple.assign();

    for (int i=0; i<numInstances; i++) {
      simpleAssignments[i] = clusterAssignments[i];
      // restore assignments to state before E-step
      clusterAssignments[i] = oldAssignments[i];
    }

    // number of differences between default and RMN assignments
    int numDiff = 0;
    int numSame = 0;
    boolean invalidAssignments = false;
    double ratioMissassigned = 0;
    double ratioNonMissassigned = 0; 

    // Make new cluster assignments using RMN inference
    for (int i = 0; i < numInstances; i++) {
      Variable var = vars[i];
      int newAssignment = var.getInfValue();
      if (verbose) {
	System.out.println("Variable " + i + " has MPE " + newAssignment);
      }

      if (clusterAssignments[i] != newAssignment) {
	if (verbose) {
	  System.out.println("Moving instance " + i + " from cluster " + clusterAssignments[i] + " to cluster " + newAssignment);
	}
	clusterAssignments[i] = newAssignment;
	moved++;
      }

      if (clusterAssignments[i] == -1)  { // current cluster assignment invalid
	invalidAssignments = true;
	break; // exit for loop
      }
    }

    // 0/NaN in RMN, fallback to SimpleAssigner
    if (invalidAssignments) {
      System.out.println("Instances not correctly assigned by RMN, backing off and assigning by SimpleAssigner");
      for (int i=0; i<numInstances; i++) {
	clusterAssignments[i] = simpleAssignments[i];
      }
    } else {    // compare RMNAssignments to simpleAssignments
      for (int i = 0; i < numInstances; i++) {
	if (clusterAssignments[i] != simpleAssignments[i]) {
	  numDiff++;
	  
	  // count number of constraint violations for this point
	  HashMap instanceConstraintHash = m_clusterer.getInstanceConstraintsHash();
	  int numViolated = 0;
	  int numTotal = 0; 
	  Object list =  instanceConstraintHash.get(new Integer(i));
	  if (list != null) {   // there are constraints associated with this instance
	    ArrayList constraintList = (ArrayList) list;
	    numTotal = constraintList.size();
	    for (int j = 0; j < constraintList.size(); j++) {
	      InstancePair pair = (InstancePair) constraintList.get(j);
	      int firstIdx = pair.first;
	      int secondIdx = pair.second;
	      
	      int centroidIdx = (firstIdx == i) ? clusterAssignments[firstIdx] : clusterAssignments[secondIdx];
	      int otherIdx = (firstIdx == i) ? clusterAssignments[secondIdx] : clusterAssignments[firstIdx];
	      
	      // check whether the constraint is violated
	      if (otherIdx != -1 && otherIdx < numClusters) { 
		if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) { 
		  numViolated++;
		} else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) { 
		  numViolated++;
		}
	      }
	    }
	  }
	  //	  System.out.println("#constraints violated for point " + i + " = " + numViolated);

	  // compare to simpleAssignments
	  double ratio = (numTotal == 0) ? 0 : ((numViolated+0.0)/numTotal);
	  System.out.print("Point: " + i + "...");
	  System.out.println("clusterAssignments: " + clusterAssignments[i] + ", simpleAssignments[i] = " + simpleAssignments[i]);
	  System.out.println("numTotal: " + numTotal + ", ratio: " + ratio);
	  if (numTotal > 0) { 
	    if (clusterAssignments[i] != simpleAssignments[i]) {
	      numDiff++;
	      System.out.println("MISSASSIGNED; violated/total = " + numViolated + "/" + numTotal + "\t=" + ((float) ratio));
	      ratioMissassigned += ratio; 
	    } else {
	      System.out.println("NOT MISASSIGNED; violated/total = " + numViolated + "/" + numTotal + "\t=" + ((float) ratio));
	      ratioNonMissassigned += ratio;
	      numSame++;
	    }
	  }
	}
      }
    }
    
    System.out.println("\tAVG for misassigned:  " + ((float) (ratioMissassigned/numDiff)) +
		       "\n\tAVG for non-misassigned:  " + ((float) (ratioNonMissassigned/numSame)));
    System.out.println("Moved " + moved + " points in RMN inference E-step");

    
    /****/
    /**** End of comparing to default assigner */
    /****/
    
    return moved;
  }

  /**
   * Get/Set m_singlePass
   * @param b truth value
   */
  public void setSinglePass (boolean b) {
    this.m_singlePass = b;
  }
  public boolean getSinglePass () {
    return  m_singlePass;
  }

  /**
   * Get/Set m_expScalingFactor
   * @param s scaling factor
   */
  public void setExpScalingFactor (double s) {
    this.m_expScalingFactor = s;
    System.out.println("Setting expScalingFactor to: " + m_expScalingFactor);
  }
  public double getExpScalingFactor () {
    return  m_expScalingFactor;
  }

  /**
   * Get/Set m_constraintWeight
   * @param w weight
   */
  public void setConstraintWeight (double w) {
    this.m_constraintWeight = w;
    System.out.println("Setting constraintWeight to: " + m_constraintWeight);
  }
  public double getConstraintWeight () {
    return  m_constraintWeight;
  }

  public void setOptions (String[] options)
    throws Exception {
    // TODO
  }

  public Enumeration listOptions () {
    // TODO
    return null;
  }
  
  public String [] getOptions ()  {
    String[] options = new String[5];
    int current = 0;

    if (m_singlePass) {
      options[current++] = "-singlePass";
    }
//      options[current++] = "-expScale";
//      options[current++] = "" + m_expScalingFactor;
    options[current++] = "-constrWt";
    options[current++] = "" + m_constraintWeight;

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }
} 

// TODO: 
// 1. Add potential nodes for multiple-metric WeightedMahalanobis or WeightedEuclidean
