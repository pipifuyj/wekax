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
 *    LPAssigner.java
 *    LP-based assignment for K-Means following Kleinberg&Tardos
 *    Copyright (C) 2004 Misha Bilenko
 *
 */

package weka.clusterers.assigners; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.core.metrics.*;
import  weka.clusterers.*;
import  weka.clusterers.assigners.*;


import jmatlink.JMatLink;

public class LPAssigner extends MPCKMeansAssigner {
  
 


  /** fields to be initialized from m_clusterer */
  protected Instances m_instances = null;
  protected HashMap m_constraintHash = null; 

  protected int m_numInstances = 0;
  protected int m_numClusters = 0;
  protected int m_numConstraints = 0;
  protected int m_numCLConstraints = 0;
  protected int m_numMLConstraints = 0;

  protected int m_numLabelVars = 0;
  protected int m_numConstraintVars = 0;
  protected int m_numVars = 0; 
  
  protected boolean m_useMultipleMetrics = false;
  protected Metric m_metric = null;
  protected LearnableMetric[] m_metrics = null;
  protected double[] m_maxCLDistances = null; 
  protected Instances m_centroids = null;


  /** Different engines that can be used to solve the LP */
  public static final int ENGINE_JMATLINK = 1;
  public static final int ENGINE_OCTAVE = 2;
  public static final int ENGINE_MATLAB = 4;
  public static final int ENGINE_TOMLAB = 8;
  public static final Tag[] TAGS_ENGINE_TYPE = {
    new Tag(ENGINE_JMATLINK, "Matlab via JMatLink"),
    new Tag(ENGINE_OCTAVE, "Octave"),
    new Tag(ENGINE_MATLAB, "Matlab"),
    new Tag(ENGINE_TOMLAB, "TomLab via Matlab")
      };
  /** The engine*/
  protected int m_engineType = ENGINE_MATLAB;

  /** The matlab engine  */
  protected JMatLink m_engine = null;

  /** Engine auxiliary files */

  /** Path to the directory where temporary files will be stored */
  protected String m_tempDirPath = new String("/tmp/");
  protected File m_tempDirFile = null;
  
  protected String m_progFilename = new String(m_tempDirPath + "LPAssigner.m");
  protected String m_dataFilenameBase = new String("data");
  protected String m_dataFilename = null; 

  protected String m_outFilenameBase = new String("output");
  protected String m_outFilename = null; 

  
 /** This is a sequential assignment method */
  public boolean isSequential() {
    return false;
  }
  
  /** Initialize fields from the current clustererer */
  protected void initialize() throws Exception {
    if (m_clusterer != null) {
      m_instances = m_clusterer.getInstances();
      m_numInstances = m_instances.numInstances();
      
      m_constraintHash = m_clusterer.getConstraintsHash();
      m_numConstraints = m_constraintHash.size();
      m_numMLConstraints = 0;
      m_numCLConstraints = 0;
      // go through the constraints and count ML and CL
      Iterator pairItr = ((Set) m_constraintHash.keySet()).iterator();
      while(pairItr.hasNext()) {
	InstancePair pair = (InstancePair) pairItr.next();
	int linkType = ((Integer) m_constraintHash.get(pair)).intValue();
	if (linkType == InstancePair.MUST_LINK) {
	  m_numMLConstraints++;
	} else if (linkType == InstancePair.CANNOT_LINK) {
	  m_numCLConstraints++;
	}
      }
      System.out.println(m_numConstraints +" total constraints:  " + m_numMLConstraints + " must-links and " + m_numCLConstraints +
			 " cannot-links"); 
      
      m_numClusters = m_clusterer.getNumClusters();
      m_useMultipleMetrics = m_clusterer.getUseMultipleMetrics();
      m_metric = m_clusterer.getMetric();
      m_metrics = m_clusterer.getMetrics();
      m_centroids = m_clusterer.getClusterCentroids();

      if (m_clusterer.m_maxCLPoints != null) { 
	m_maxCLDistances = calculateMaxDistances(m_clusterer.m_maxCLPoints);
      }
    } else {
      System.err.println("\n******Clusterer is null in LPAssigner.initialize()!\n******"); 
    }
  }
  
  /** The main method
   *  @return the number of points that changed assignment
   */
  public int assign() throws Exception {
    int moved = 0;

    initialize();
    
    // open the engine
    if (m_engineType == ENGINE_JMATLINK) { 
      if (m_engine == null) {
	m_engine = new JMatLink();
      }
      m_engine.engOpen();
    }

    /** formulate the LP **/
    
    // Coefficients of the objective function. Consist of the following:
    // 1) distortion coeffs x_{ij} - indexed as currCluster*numInstances+currInstance;
    //    x_{ij}=1 iff i-th instance belongs to j-th cluster
    // 2) constraint coeffs WRT cluster j - indexed as currConstraint*numClusters+currCluster
    //    y_{ij}=1 iff i-th constraint is violated and either 1st or 2nd instance belongs to j-th cluster
    m_numLabelVars = m_numInstances * m_numClusters;
    m_numConstraintVars = m_numConstraints * m_numClusters;
    m_numVars = m_numLabelVars + m_numConstraintVars;
    System.out.println("m_numLabelVars=" + m_numLabelVars + "\tm_numConstraintVars=" + m_numConstraintVars); 
    double [] objCoeffs = new double[m_numVars]; 

    accumulateDistortionCoeffs(objCoeffs);
    accumulateConstraintCoeffs(objCoeffs);

    // create the array of equality constraints (sum of probs for each instance is 1)
    double[][] A_eq = new double[m_numInstances][m_numVars];
    for (int instanceIdx = 0; instanceIdx < m_numInstances; instanceIdx++) {
      for (int clusterIdx = 0; clusterIdx < m_numClusters; clusterIdx++) {
	A_eq[instanceIdx][clusterIdx * m_numInstances + instanceIdx] = 1;
      }
    }
    double[] b_eq = new double[m_numInstances];
    for (int instanceIdx = 0; instanceIdx < m_numInstances; instanceIdx++) {
      b_eq[instanceIdx] = 1;
    }
    
    // create the array of inequality constraints (positivity + 2perML + 2perCL)
    System.out.println("allocating for A: " + (m_numVars + 2*m_numConstraints*m_numClusters) +
		       "x" + m_numVars + " (numConstraints=" + m_numConstraints);
    double[][] A = new double[m_numVars + 2*m_numMLConstraints*m_numClusters + 2*m_numCLConstraints*m_numClusters][m_numVars];
    System.out.println("done allocating for A: " + A.length + "x" + A[0].length);
    double [] b = new double[m_numVars + 2*m_numMLConstraints*m_numClusters + 2*m_numCLConstraints*m_numClusters];
    // positivity
    for (int i = 0; i < m_numVars; i++) {
      A[i][i] = -1;
      b[i] = 0;
    }
    // Constraint vars
    Iterator pairItr = ((Set) m_constraintHash.keySet()).iterator();
    int idx = 0;
    int offset = m_numVars;
    while(pairItr.hasNext()) {
      InstancePair pair = (InstancePair) pairItr.next();
      int linkType = ((Integer) m_constraintHash.get(pair)).intValue();
      if (linkType == InstancePair.MUST_LINK) { 
	for (int centroidIdx = 0; centroidIdx < m_numClusters; centroidIdx++) { 
	  A[offset+2*idx*m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.first] = 1;
	  A[offset+2*idx*m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.second] = -1;
	  A[offset+2*idx*m_numClusters + centroidIdx][m_numLabelVars + idx * m_numClusters + centroidIdx] = -1;

	  A[offset+2*idx*m_numClusters + m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.first] = -1;
	  A[offset+2*idx*m_numClusters + m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.second] = 1;
	  A[offset+2*idx*m_numClusters + m_numClusters + centroidIdx][m_numLabelVars + idx * m_numClusters + centroidIdx] = -1;
	}
      } else  if (linkType == InstancePair.CANNOT_LINK) {
	for (int centroidIdx = 0; centroidIdx < m_numClusters; centroidIdx++) {
	  A[offset+2*idx*m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.first] = -1;
	  A[offset+2*idx*m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.second] = -1;
	  A[offset+2*idx*m_numClusters + centroidIdx][m_numLabelVars + idx * m_numClusters + centroidIdx] = 1;

	  A[offset+2*idx*m_numClusters + m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.first] = 1;
	  A[offset+2*idx*m_numClusters + m_numClusters + centroidIdx][centroidIdx * m_numInstances + pair.second] = 1;
	  A[offset+2*idx*m_numClusters + m_numClusters + centroidIdx][m_numLabelVars + idx * m_numClusters + centroidIdx] = -1;
	  b[offset+2*idx*m_numClusters + m_numClusters + centroidIdx] = 1;
	}
      }
      idx++;
    }
    

    /** Send the LP to the engine and get back the solution **/
    
    double[][] probs = null; 
    if (m_engineType == ENGINE_OCTAVE || m_engineType == ENGINE_MATLAB || m_engineType == ENGINE_TOMLAB ) {
      dumpData(objCoeffs, A_eq, b_eq, A, b);
      prepareEngine();
      runEngine();
      probs = getSolution();
    } else if (m_engineType == ENGINE_JMATLINK) { 
      m_engine.engPutArray("f", objCoeffs);
      m_engine.engPutArray("Aeq", A_eq);
      m_engine.engPutArray("beq", b_eq);
      m_engine.engPutArray("A", A);
      m_engine.engPutArray("b", b);

      // solve the LP
      m_engine.engEvalString("x = linprog(f,A,b,Aeq,beq)");

      // get the solution back 
      probs = m_engine.engGetArray("x");
      m_engine.engClose();
    } else {
      throw new Exception("Unknown engine type: " + m_engineType);
    }       

    if (m_clusterer.getVerbose()) { 
      for (int i = 0; i < probs.length; i++) {
	for (int j = 0; j < probs[i].length; j++) {
	  System.out.print(((float)probs[i][j]) + "\t");
	}
      }
    }


    /** Get cluster assignments from the solution probabilistically */
    int [] assignments = new int [m_numInstances];
    Arrays.fill(assignments, -1);
    int numAssigned = 0;
    Random r = new Random(m_clusterer.getRandomSeed());
    int phase = 0;
    int m_maxPhases = 5000;

    while (numAssigned < m_numInstances && phase < m_maxPhases) {
      // pick a random label
      int clusterIdx = r.nextInt(m_numClusters);
      double alpha = r.nextDouble();
      
      for (int i = 0; i < m_numInstances; i++) {
	if (assignments[i] == -1) {
	  if (probs[clusterIdx * m_numInstances + i][0] >= alpha) {
	    assignments[i] = clusterIdx;
	    numAssigned++; 
	  } 
	} 
      }
      phase++; 
    }


    /****/
    /**** Compare to default assigner */
    /****/
    
    SimpleAssigner simple = new SimpleAssigner(m_clusterer);
    int [] clusterAssignments = m_clusterer.getClusterAssignments();
    int [] oldAssignments = new int[m_numInstances];
    int [] simpleAssignments = new int[m_numInstances];

    // backup assignments before E-step
    for (int i = 0; i < m_numInstances; i++) {
      oldAssignments[i] = clusterAssignments[i];
    }

    // get assignments with default E-step
    simple.assign();
    for (int i = 0; i < m_numInstances; i++) {
      simpleAssignments[i] = clusterAssignments[i];
      // restore assignments to state before E-step
      clusterAssignments[i] = oldAssignments[i];
    }

    // number of differences between default and RMN assignments
    int numDiff = 0;
    int numSame = 0;
    int totalDiff = 0; 
    boolean invalidAssignments = false;

    // Make new cluster assignments, count num moved
    System.out.println(phase + " phases; " + numAssigned + "/" + m_numInstances + " assigned"); 
    double ratioMissassigned = 0;
    double ratioNonMissassigned = 0; 
    for (int i = 0; i < m_numInstances; i++) {
      if (clusterAssignments[i] != assignments[i]) {
//  	System.out.println("Moving instance " + i + " from cluster " + clusterAssignments[i] + " to cluster " + assignments[i]);
	clusterAssignments[i] = assignments[i];
	moved++;
      }

      

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
	  if (otherIdx != -1 && otherIdx < m_numClusters) {
	    if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) { 
	      numViolated++;
	    } else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) { 
	      numViolated++;
	    }
	  }
	}
      }

      // compare to simpleAssignments
      if (clusterAssignments[i] != simpleAssignments[i]) {
	totalDiff++;
      }
      double ratio = (numTotal == 0) ? 0 : ((numViolated+0.0)/numTotal);
      if (numTotal > 0) { 
	if (clusterAssignments[i] != simpleAssignments[i]) {
	  numDiff++;
//  	  System.out.println("MISSASSIGNED; violated/total = " + numViolated + "/" + numTotal + "\t=" + ((float) ratio));


	  // check where it would be assigned without taking constraints into account:
	  // KLUDGE-ish, assuming a single metric
	  double closestDistance = Double.MAX_VALUE;
	  int centroidIdx = -1;
	  Instance instance = m_instances.instance(i); 
	  for (int j = 0; j < m_numClusters; j++) { 
	    Instance centroid = m_clusterer.getClusterCentroids().instance(j);
	    double distance = m_clusterer.getMetric().distance(centroid, instance);
	    if (distance < closestDistance) {
	      closestDistance = distance;
	      centroidIdx = j;
	    }
	  }

	  System.out.println("ASSIGNED to: " + clusterAssignments[i] + "; SimpleAssigner assigns to: " +
			     simpleAssignments[i] + "; without constraints closest centroid: " + centroidIdx);
	  
	  
	  ratioMissassigned += ratio; 
	} else {
//  	  System.out.println("NOT MISASSIGNED; violated/total = " + numViolated + "/" + numTotal + "\t=" + ((float) ratio));
	  ratioNonMissassigned += ratio;
	  numSame++;
	}
      }
    }
    System.out.println("Total missassigned: " + totalDiff); 
    System.out.println("\tAVG for misassigned:  " + ((float) (ratioMissassigned/numDiff)) +
		       "\n\tAVG for non-misassigned:  " + ((float) (ratioNonMissassigned/numSame)));
    System.out.println("Moved " + moved + " points in RMN inference E-step");

    
    /****/
    /**** End of comparing to default assigner */
    /****/

   

    
    return moved;
  }


  /** go through all instances and all clusters and accumulate the distortion contributions */ 
  protected void accumulateDistortionCoeffs(double [] objCoeffs) throws Exception { 

    for (int centroidIdx = 0; centroidIdx < m_numClusters; centroidIdx++) {
      Instance centroid = m_centroids.instance(centroidIdx);

      for (int instanceIdx = 0; instanceIdx < m_numInstances; instanceIdx++) {
	Instance instance = m_instances.instance(instanceIdx);
	int coeffIdx = centroidIdx * m_numInstances + instanceIdx;

	if (!m_clusterer.isObjFunDecreasing()) { // increasing obj. function
	  if (m_useMultipleMetrics) { // multiple metrics
	    objCoeffs[coeffIdx] = m_metrics[centroidIdx].similarity(instance, centroid);
	  } else {
	    objCoeffs[coeffIdx] = m_metric.similarity(instance, centroid);
	  }
	} else { // decreasing obj. function
	  if (m_useMultipleMetrics) { // multiple metrics
	    objCoeffs[coeffIdx] = m_metrics[centroidIdx].distance(instance, centroid);
	  } else {
	    objCoeffs[coeffIdx] = m_metric.distance(instance, centroid);
	  }
	}
      }
    }
  }


  /** Accumulate contribution from constraints */
  protected void accumulateConstraintCoeffs(double [] objCoeffs) throws Exception{
    if (m_constraintHash != null) {
      Set pointPairs = (Set) m_constraintHash.keySet(); 
      Iterator pairItr = pointPairs.iterator();
      int idx = 0;

      while( pairItr.hasNext() ){
	InstancePair pair = (InstancePair) pairItr.next();
	addPairPenalties(pair, idx, objCoeffs);
	idx++; 
      }
    }
  }
	



  /** accumulate penalties associated with a given constraint */
  protected void addPairPenalties(InstancePair pair, int idx, double[] objCoeffs) throws Exception { 
    int instance1Idx = pair.first;
    int instance2Idx = pair.second;
    Instance instance1 = m_instances.instance(instance1Idx);
    Instance instance2 = m_instances.instance(instance2Idx);
    int linkType = ((Integer) m_constraintHash.get(pair)).intValue();

    double cost = 0;
    if (linkType == InstancePair.MUST_LINK) {
      cost = m_clusterer.getMustLinkWeight();
    } else if (linkType == InstancePair.CANNOT_LINK) {
      cost = m_clusterer.getCannotLinkWeight();
    }

    // if a single metric is used, we don't need to calculate separately for each cluster
    if (!m_useMultipleMetrics) {  // MAJOR KLUDGE.  TODO:  create penalty(InstancePair) method in MPCKMeans; use both internally and here;
                                  // avoid iterating through constraints inside individual calculateConstraintPenalties methods
      double penalty = 0;

      // add the penalty for different types of metrics
      if (m_metric instanceof WeightedDotP) {
	double sim = m_metric.similarity(instance1, instance2);
	if (linkType == InstancePair.MUST_LINK) {
	  penalty = -cost * (1 - sim);
	} else if (linkType == InstancePair.CANNOT_LINK) {
	  penalty = -cost * sim;
	}
      } else if (m_metric instanceof KL) {
	double distance = ((KL) m_metric).distanceJS(instance1, instance2);
	if (linkType == InstancePair.MUST_LINK) {
	  penalty = cost * distance;
	} else if (linkType == InstancePair.CANNOT_LINK) { 
	  penalty = cost * (2.0 - distance); 
	}
      } else if (m_metric instanceof WeightedEuclidean || m_metric instanceof WeightedMahalanobis) {
	double distance = m_metric.distance(instance1, instance2);
	if (linkType == InstancePair.MUST_LINK) {
	  penalty = cost * distance * distance;
	} else if (linkType == InstancePair.CANNOT_LINK) {
	  penalty = cost * (m_maxCLDistances[0] * m_maxCLDistances[0] - distance * distance); 
	} 
      } else {
	throw new Exception("Unknown metric: " + m_metric.getClass().getName());
      }

      // y_m = 0.5 sum_j (y_{mj})
      if (linkType == InstancePair.MUST_LINK) {
	penalty = 0.5 * penalty; 
      } else {
	//	penalty = -0.5 * penalty;
      }

      int offset = m_numLabelVars;
      for (int centroidIdx = 0; centroidIdx < m_numClusters; centroidIdx++) {
	objCoeffs[offset + idx * m_numClusters + centroidIdx] += penalty;
      }
      
    } else { // MULTIPLE METRICS   // KLUDGE - TODO - CURRENTLY WRONG!
//        for (int centroidIdx1 = 0; centroidIdx1 < m_numClusters; centroidIdx1++) {
//  	for (int centroidIdx2 = 0; centroidIdx2 < m_numClusters; centroidIdx2++) {
//  	  double penalty = 0;

//  	  if (m_metric instanceof WeightedDotP) {
//  	    double sim1 = m_metrics[centroidIdx1].similarity(instance1, instance2);
//  	    double sim2 = m_metrics[centroidIdx2].similarity(instance1, instance2);
//  	    penalty = 0.5 * cost * (1 - sim2) + 0.5 * cost * (1 - sim1);
//  	  } else if (m_metric instanceof KL) {
//  	    double penalty1 = ((KL) m_metrics[centroidIdx1]).distanceJS(instance1, instance2);
//  	    double penalty2 = ((KL) m_metrics[centroidIdx2]).distanceJS(instance1, instance2);
//  	    penalty = 0.5 * cost * (penalty1 + penalty2);
//  	  } else if (m_metric instanceof WeightedEuclidean || m_metric instanceof WeightedMahalanobis) {
//  	    double distance1 = m_metrics[centroidIdx1].distance(instance1, instance2);
//  	    double distance2 = m_metrics[centroidIdx2].distance(instance1, instance2);
//  	    penalty = 0.5 * cost * (distance1*distance1 + distance2*distance2);
//  	  } else {
//  	    throw new Exception("Unknown metric: " + m_metric.getClass().getName());
//  	  }

//  	  objCoeffs[centroidIdx1 * m_numInstances + instance1Idx] += penalty;
//  	  objCoeffs[centroidIdx1 * m_numInstances + instance2Idx] += penalty;
//  	  objCoeffs[centroidIdx2 * m_numInstances + instance1Idx] += penalty;
//  	  objCoeffs[centroidIdx2 * m_numInstances + instance2Idx] += penalty;
//  	}
//        }
    }
  }

  /**
   * Dump data matrix into a file
   */
  protected void dumpData(double[] objCoeffs, double[][] A_eq, double[] b_eq, double[][] A, double[] b) {
    if (m_engineType == ENGINE_TOMLAB) {
      dumpDataTomLab(objCoeffs,A_eq, b_eq, A,b);
    } else { 
      try {
	File dataFile = File.createTempFile(m_dataFilenameBase, ".m", m_tempDirFile);
	m_dataFilename = dataFile.getPath();
	if (!m_clusterer.getVerbose()) { 
	  dataFile.deleteOnExit();
	}

	PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(dataFile)));

	// dump f
	writer.print("f = [");
	for (int i = 0; i < objCoeffs.length; i++) {
	  writer.print(objCoeffs[i] + "; ");
	}
	writer.println("];");

	// dump Aeq

	if (m_engineType != ENGINE_OCTAVE) {
	  writer.print("Aeq = [");
	  for (int i = 0; i < A_eq.length; i++) {
	    for (int j = 0; j < A_eq[i].length; j++) {
	      writer.print(A_eq[i][j] + ", ");
	    }
	    writer.flush();
	    writer.println(";"); 
	  }
	  writer.println("];");
	} else { // for octave, we dump into a separate file...
	  PrintWriter writerAeq = new PrintWriter(new BufferedOutputStream(new FileOutputStream(m_tempDirPath + "Aeq")));
	  for (int i = 0; i < A_eq.length; i++) {
	    for (int j = 0; j < A_eq[i].length; j++) {
	      writerAeq.print(A_eq[i][j] + " ");
	    }
	    writerAeq.flush();
	    writerAeq.println();
	  }
	  writerAeq.close();
	} 
	
	// dump b
	writer.print("beq = [");
	for (int i = 0; i < b_eq.length; i++) {
	  writer.print(b_eq[i] + "; ");
	}
	writer.println("];");

	// dump A
	PrintWriter writerA = new PrintWriter(new BufferedOutputStream(new FileOutputStream(m_dataFilename + ".A")));
	for (int i = 0; i < A.length; i++) {
	  for (int j = 0; j < A[i].length; j++) {
	    writerA.print(A[i][j] + " ");
	  }
	    writerA.println();
	}
	writerA.flush();
	writerA.close();

	// dump b
	writer.print("b = [");
	for (int i = 0; i < b.length; i++) {
	  writer.print(b[i] + "; ");
	}
	writer.println("];");
      
	writer.close();
      } catch (Exception e) {
	System.err.println("Could not create temporary file \'" + m_dataFilename + "\' for dumping the LP: " + e);
      }
    }
  }

/**
   * Dump data matrix into a file
   */
  protected void dumpDataTomLab(double[] objCoeffs, double[][] A_eq, double[] b_eq, double[][] A, double[] b) {
    try {
      File dataFile = File.createTempFile(m_dataFilenameBase, ".m", m_tempDirFile);
      m_dataFilename = dataFile.getPath();
      if (!m_clusterer.getVerbose()) { 
	dataFile.deleteOnExit();
      }

      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(dataFile)));

      // dump f
      writer.print("f = [");
      for (int i = 0; i < objCoeffs.length; i++) {
	writer.print(objCoeffs[i] + "; ");
      }
      writer.println("];");

      // dump xl and xu
      writer.println("xl = zeros(" + m_numVars + ",1);");
      writer.println("xu = ones(" + m_numVars + ",1);");

      // dump bu
      writer.print("bu = [ones(" + m_numInstances + ",1); ");
      for (int i = m_numVars; i < b.length; i++) {
	writer.print(b[i] + "; ");
      }
      writer.println("];");

      // dump bl 
      writer.println("bl = [ones(" + m_numInstances + ",1); zeros(" + b.length + "-" + m_numVars + ",1)];");
      writer.println("bl(" + (m_numInstances+1) + ":" + (m_numInstances +  b.length - m_numVars) + ",:)=-Inf;"); 

      writer.close();

      // dump A into a separate file
      PrintWriter writerA = new PrintWriter(new BufferedOutputStream(new FileOutputStream(m_dataFilename + ".A")));
      File aFile = new File(m_dataFilename + ".A");
      aFile.deleteOnExit();

      // first, dump Aeq
      for (int i = 0; i < A_eq.length; i++) {
	for (int j = 0; j < A_eq[i].length; j++) {
	  writerA.print(A_eq[i][j] + " ");
	}
	writerA.flush();
	writerA.println();
      }

      // next, dump constraints from A
      for (int i = m_numVars; i < A.length; i++) {
	for (int j = 0; j < A[i].length; j++) {
	  writerA.print(A[i][j] + " ");
	}
	writerA.flush();
	writerA.println();
      }
      writerA.close();
    } catch (Exception e) {
      System.err.println("Could not create temporary file \'" + m_dataFilename + "\' for dumping the LP: " + e);
    }
  }
  

  /** Read the solution from the output file of Octave */
protected double[][] getSolution() {
    double[][] probs = new double[m_numLabelVars][1]; 

    try { 
      BufferedReader r = new BufferedReader(new FileReader(m_outFilename));
      String s = null;
      int i = 0; 
      while ((s = r.readLine()) != null && i < m_numLabelVars) {
	probs[i++][0] = Double.parseDouble(s);
      }
    } catch (Exception e) {
      System.out.println("Problems reading the solution from the engine: " + e);
      e.printStackTrace();
    }
    File aFile = new File(m_dataFilename + ".A");
    aFile.delete();
    File dataFile = new File(m_dataFilename);
    dataFile.delete();
    return probs; 
  } 

  
  /** Create octave m-file 
   * @param filename file where the script is created
   */
  public void prepareEngine() {
    try{
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(m_progFilename)));
      writer.println("cd " + m_tempDirPath + ";");
      String dataFilename =  Utils.removeSubstring(m_dataFilename, m_tempDirPath);
      dataFilename =  Utils.removeSubstring(dataFilename, ".m");
      writer.println(dataFilename + ";");

      switch (m_engineType) { 
      case ENGINE_MATLAB:
	writer.println("A = load(\'" + m_dataFilename + ".A" + "\');");
	writer.println("x = linprog(f,A,b,Aeq,beq);");
	break;
      case ENGINE_TOMLAB:
	writer.println("cd /u/ml/software/tomlab;");
	writer.println("startup;"); 
	writer.println("A = load(\'" + m_dataFilename + ".A" + "\');");
	writer.println("Prob = lpAssign(f,A,bl,bu,xl,xu,[],'test');");
	writer.println("Result = tomRun('pdco', Prob,[],1);");
	writer.println("x = Result.x_k;");
	break;
      case ENGINE_OCTAVE: // load A and Aeq stored in auxiliary files
	writer.println("load A;");
	writer.println("load Aeq;");
      }
      

      File outFile = File.createTempFile(m_outFilenameBase, ".out", m_tempDirFile);
      m_outFilename = outFile.getPath();
      if (!m_clusterer.getVerbose()) { 
	outFile.deleteOnExit();
	File outFileDump = new File(m_outFilename + ".dump");
	outFileDump.deleteOnExit();
      }

      writer.println("x");
      writer.println("save " + m_outFilename + " x -ascii;");
      writer.close();
    } 
    catch (Exception e) {
      System.err.println("Could not create script file \'" + m_progFilename + "\': " + e);
    }
  }

  
  /** Run octave in command line with a given argument
   * @param inFile file to be input 
   * @param outFile file where results are stored
   */
  public int runEngine() {
    int exitValue = -1;
    try {
      String cmd = "";
      if (m_engineType == ENGINE_OCTAVE) { 
	cmd = "octave  " + m_progFilename + " > " + m_outFilename;
      } else if (m_engineType == ENGINE_MATLAB || m_engineType == ENGINE_TOMLAB) {
	cmd = "matlab -nodesktop  -nosplash < " + m_progFilename + " > " + m_outFilename + ".dump";
      }
      System.out.println("Starting to run engine  " + m_engineType + cmd);
      Process proc = Runtime.getRuntime().exec(cmd);
      System.out.println("Waiting for process ...");

      // read the error
      if (proc != null){
	BufferedReader procError  = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
	try {
	  String line;
	  while ((line = procError.readLine()) != null){
	    System.out.println("ERROR:  " + line);	    
	  }
	} catch (Exception e) {
	  System.err.println("Problems trapping error stream in debug mode: " + e);
	  e.printStackTrace();
	}
      }

      // read the output
      if (proc != null){
	BufferedReader procOutput  = new BufferedReader(new InputStreamReader(proc.getInputStream()));
	try {
	  String line;
	  while ((line = procOutput.readLine()) != null){
	    System.out.println("OUTPUT:  " + line);
	  }
	} catch (Exception e) {
	  System.err.println("Problems trapping output in debug mode: " + e);
	  e.printStackTrace();
	}
      }

      exitValue = proc.waitFor();
      System.out.println("End of running engine, exitValue = " + exitValue);
    }
    catch (Exception e) {
      System.err.println("Problems running engine: " + e);
      e.printStackTrace();
    }
    return exitValue; 
  }

  protected double[] calculateMaxDistances(Instance maxCLPoints[][]) throws Exception {
    double []  maxCLDistances = new double[maxCLPoints.length];
    for (int i = 0; i < maxCLDistances.length; i++) {
      if (m_useMultipleMetrics) { 
	maxCLDistances[i] = m_metrics[i].distance(maxCLPoints[i][0],
						  maxCLPoints[i][1]);
      } else {
	maxCLDistances[i] = m_metric.distance(maxCLPoints[0][0],
					      maxCLPoints[0][1]);
      }
    }
    return maxCLDistances; 
  } 

  /** Set the engine type
   * @param type one of the kernel types */
  public void setEngineType(SelectedTag engineType) {
    if (engineType.getTags() == TAGS_ENGINE_TYPE) {
      m_engineType = engineType.getSelectedTag().getID();
    }
  }
  
  /** Get the engine type
   * @return engine type  */
  public SelectedTag getEngineType() {
    return new SelectedTag(m_engineType, TAGS_ENGINE_TYPE);
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
    String[] options = new String[1];
    int current = 0;
    
    switch (m_engineType) {
    case ENGINE_JMATLINK:
      options[current++] = "jmatlink";
      break;
    case ENGINE_OCTAVE:
      options[current++] = "octave";
      break;
    case ENGINE_MATLAB:
      options[current++] = "matlab";
      break;
    case ENGINE_TOMLAB:
      options[current++] = "tomlab";
      break;
    default:
      options[current++] = "unknown";
    }
    return options;
  }
} 



