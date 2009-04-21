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
 *    WEuclideanLearner.java
 *    Copyright (C) 2004 Mikhail Bilenko and Sugato Basu
 *
 */

package weka.clusterers.metriclearners; 

import java.util.*;

import weka.core.*;
import weka.core.metrics.*;
import weka.clusterers.MPCKMeans;
import weka.clusterers.InstancePair;


/** 
 * A closed-form learner for WeightedEuclidean
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu
 * @version $Revision: 1.5 $ */

public class WEuclideanLearner extends  MPCKMeansMetricLearner {
  
  public void resetLearner() {
  } 

  
  /** if clusterIdx is -1, all instances are used
   * (a single metric for all clusters is used) */   
  public boolean trainMetric(int clusterIdx) throws Exception {
    Init(clusterIdx); 

    double[] weights = new double[m_numAttributes];
    int violatedConstraints = 0;
    int numInstances = 0; 

    for (int instIdx = 0; instIdx < m_instances.numInstances(); instIdx++) {
      int assignment = m_clusterAssignments[instIdx];

      // only instances assigned to this cluster are of importance
      if (assignment == clusterIdx || clusterIdx == -1) {
	numInstances++;
	if (clusterIdx < 0) {
	  m_centroid = m_kmeans.getClusterCentroids().instance(assignment); 
	} 
	
	// accumulate variance
	Instance instance = m_instances.instance(instIdx);
	Instance diffInstance = m_metric.createDiffInstance(instance, m_centroid); 
	for (int attr = 0; attr < m_numAttributes; attr++) {
	  weights[attr] += diffInstance.value(attr); 
	}

	// check all constraints for this instance
	Object list =  m_instanceConstraintMap.get(new Integer(instIdx));
	if (list != null) {   // there are constraints associated with this instance
	  ArrayList constraintList = (ArrayList) list;
	  for (int i = 0; i < constraintList.size(); i++) {
	    InstancePair pair = (InstancePair) constraintList.get(i);
	    int linkType = pair.linkType;
	    int firstIdx = pair.first;
	    int secondIdx = pair.second;
	    Instance instance1 = m_instances.instance(firstIdx);
	    Instance instance2 = m_instances.instance(secondIdx);
	    int otherIdx = (firstIdx == instIdx) ? m_clusterAssignments[secondIdx]
	      : m_clusterAssignments[firstIdx];

	    if (otherIdx != -1) {  // check whether the constraint is violated
	      if (otherIdx != assignment && linkType == InstancePair.MUST_LINK) {
		diffInstance = m_metric.createDiffInstance(instance1, instance2);
		for (int attr = 0; attr < m_numAttributes; attr++) {  
		  weights[attr] += 0.5 * m_MLweight * diffInstance.value(attr);
		}
	      }
	      else if (otherIdx == assignment && linkType == InstancePair.CANNOT_LINK){ 
		diffInstance = m_metric.createDiffInstance(instance1, instance2);
		for (int attr = 0; attr < m_numAttributes; attr++) {
		  // this constraint will be counted twice, hence 0.5
		  weights[attr] += 0.5 * m_CLweight * m_maxCLDiffInstance.value(attr);
		  weights[attr] -= 0.5 * m_CLweight * diffInstance.value(attr); 
		}
	      }
	    } 
	  }
	}
      }
    }
//      System.out.println("Updating cluster " + clusterIdx
//  		       + " containing " + numInstances); 

    // check the weights
    double [] newWeights = new double[m_numAttributes];
    double [] currentWeights = m_metric.getWeights();

    boolean needNewtonRaphson = false;
    for (int attr = 0; attr < m_numAttributes; attr++) {
      if (weights[attr] <= 0) { // check to avoid divide by 0 - TODO!
	System.out.println("Negative weight " + weights[attr]
			   + " for clusterIdx=" + clusterIdx
			   + "; using prev value=" + currentWeights[attr]); 
	newWeights[attr] = currentWeights[attr];
	//  	needNewtonRaphson = true;
	//	break;
      } else {
	if (m_regularize) { // solution of quadratic equation - TODO!
	  int n = m_instances.numInstances();
	  double ratio =  (m_logTermWeight * n) / (2 * weights[attr]);
	  newWeights[attr] = ratio + Math.sqrt(ratio*ratio +
					       (m_regularizerTermWeight*n)
					       /weights[attr]);
	} else {
	  newWeights[attr] = m_logTermWeight * numInstances  / weights[attr];
	}
      }
    }
    
    // do NR if needed
    if (needNewtonRaphson) {
      System.out.println("GOING TO NEWTON-RAPHSON!!!\n"); 
      newWeights = updateWeightsUsingNewtonRaphson(currentWeights, weights);
    } 

    // PRINT routine
    //      System.out.println("Total constraints violated: " + violatedConstraints/2 + "; weights are:"); 
    //      for (int attr=0; attr<numAttributes; attr++) {
    //        System.out.print(newWeights[attr] + "\t");
    //      }
    //      System.out.println();
    // end PRINT routine

    m_metric.setWeights(newWeights);
    return true;
  }

    /** calculates weights using Newton Raphson, to satisfy the
      positivity constraint of each attribute weight, returns learned
      attribute weights. Note: currentAttrWeights is the inverted version
      of the current m_metric weights.
  */
  protected double [] updateWeightsUsingNewtonRaphson
    (double [] currentAttrWeights, double [] invUnconstrainedAttrWeights)
    throws Exception {
    int numAttributes = currentAttrWeights.length; 
    double [] iterAttrWeights = currentAttrWeights;
    
//      System.out.println("Updating Weights Using NewtonRaphson");
//      do {
//        // sets new attribute weights using NR with line search for alpha
//        iterAttrWeights = nrWithLineSearchForAlpha(iterAttrWeights,
//  						 invUnconstrainedAttrWeights); 
//        // set current attribute weight to m_metric, recalculate obj. fn.
//        m_OldObjective = m_Objective;
//        ((LearnableM_metric) m_m_metric).setWeights(iterAttrWeights);
//        calculateObjectiveFunction();
//      } while (!convergenceCheck(m_OldObjective, m_Objective, false)); // objective function not guaranteed to monotonically decrease across NR iterations, so don't do convergence check
    return iterAttrWeights;
  }

  /** Does one NR step, calculates the alpha (using line search) that
      does not violate positivity constraint of each attribute weight,
      returns new values of attribute weights */
  protected double [] nrWithLineSearchForAlpha(double [] currAttrWeights,
					       double [] invUnconstrainedAttrWeights)
    throws Exception {
    int numAttributes = currAttrWeights.length;
    double [] raphsonWeights = new double[numAttributes];
    double top = 1, bottom = 0, alpha = 1;
    boolean satisfiesConstraints = true;
        
//      // initial check for alpha = top
//      System.out.println("Evaluating at alpha=1");
//      for (int attr = 0; attr < numAttributes; attr++) {
//        raphsonWeights[attr] = currAttrWeights[attr] * (1 - alpha * (currAttrWeights[attr] * invUnconstrainedAttrWeights[attr] - 1));
//        if (raphsonWeights[attr] < 0) {
//  	satisfiesConstraints = false;
//  	System.out.println("Negative raphsonWeight for attr: " + attr + ", exiting loop");
//  	break;
//        }
//        //        System.out.println("Curr weights: " + currAttrWeights[attr] + ", alpha: " + alpha + ", m_Objective: " + m_Objective);
//        //        System.out.println("Raphson weights[" + attr +"] = " + raphsonWeights[attr]);
//      }

//      if (!satisfiesConstraints) {
//        // line search for alpha between bottom and top
//        // satisfiesConstraints is false at top, true at bottom
//        // we want max. alpha in [0,1] for which satisfiesConstraints is true
//        System.out.println("Starting line search for alpha");
//        while ((top-bottom) > m_NRConvergenceDifference && bottom <= top) {
//  	alpha = (bottom + top)/2;
//  	satisfiesConstraints = true;
//  	for (int attr = 0; attr < numAttributes; attr++) {
//  	  raphsonWeights[attr] = currAttrWeights[attr] * (1 - alpha * (currAttrWeights[attr] * invUnconstrainedAttrWeights[attr] - 1));
//  	  if (raphsonWeights[attr] < 0) {
//  	    satisfiesConstraints = false;
//  	    System.out.println("Negative raphsonWeight for attr: " + attr + ", exiting loop");
//  	    break;
//  	  }
//  	  //  	  System.out.println("In line search ... curr weights: " + currAttrWeights[attr] + ", alpha: " + alpha + ", m_Objective: " + m_Objective);
//  	  //  	  System.out.println("In line search ... raphson weights[" + attr +"] = " + raphsonWeights[attr]);
//  	}
//  	if (!satisfiesConstraints) {
//  	  top = alpha;
//  	} else {
//  	  bottom = alpha;
//  	}
//  	System.out.println("Top: " + top + ", Bottom: " + bottom);
//        }
//        alpha = bottom;
    
//        System.out.println("Final alpha: " + alpha + ", final objective: " + m_Objective);
//        System.out.print("Final weights: ");
//        for (int attr = 0; attr < numAttributes; attr++) {
//  	raphsonWeights[attr] = currAttrWeights[attr] * (1 - alpha * (currAttrWeights[attr] * invUnconstrainedAttrWeights[attr] - 1));
//  	System.out.print(raphsonWeights[attr] + "\t");
//        }
//        System.out.println();
//      } else {
//        System.out.println("Constraints satisfied");
//      }

    return raphsonWeights;
  }


// OLD CODE FOR MULTIPLE:

//    /** M-step of the KMeans clustering algorithm -- updates metric
//     *  weights for the individual metrics. Invoked only whe metric is trainable
//     */
//    protected boolean updateMultipleMetricWeightsEuclidean() throws Exception {
//      if (m_regularizeWeights) {
//        System.out.println("Regularized version, calling GD version of updateMultipleMetricWeightsEuclidean!");
//        updateMultipleMetricWeightsEuclideanGD();
//      }

//      int numAttributes = m_Instances.numAttributes();
//      double[][] weights = new double[m_NumClusters][numAttributes];
//      int []counts = new int[m_NumClusters]; // count how many instances are in each cluster
//      Instance diffInstance;
//      //begin debugging variance
//      boolean debugVariance = true; 
//      double[][] trueWeights = new double[m_NumClusters][numAttributes];
//      int [] majorityClasses = new int[m_NumClusters];
//      int [][] classCounts  = new int[m_NumClusters][m_TotalTrainWithLabels.numClasses()];
//      // get the majority counts
//      // NB:  m_TotalTrainWithLabels does *not* include unlabeled data, counts here are undersampled!
//      // assuming unlabeled data came from same distribution as m_TotalTrainWithLabels, counts are still valid...
//      for (int instIdx=0; instIdx<m_TotalTrainWithLabels.numInstances(); instIdx++) {
//        Instance fullInstance = m_TotalTrainWithLabels.instance(instIdx);
//        classCounts[m_ClusterAssignments[instIdx]][(int)(fullInstance.classValue())]++;
//      }
//      for (int i = 0; i < m_NumClusters; i++){
//        int majorityClass = 0;
//        System.out.print("Cluster" + i + "\t" + classCounts[i][0]);
//        for (int j = 1; j < m_TotalTrainWithLabels.numClasses(); j++) {
//  	System.out.print("\t" + classCounts[i][j]);
//  	if (classCounts[i][j] > classCounts[i][majorityClass]) {
//  	  majorityClass = j;
//  	}
//        }
//        System.out.println();
//        majorityClasses[i] = majorityClass;
//      }
//      class MajorityChecker {
//        int [] m_majorityClasses  = null; 
//        public MajorityChecker(int [] majClasses) { m_majorityClasses = majClasses;}
//        public  boolean belongsToMajority(Instances instances, int instIdx, int centroidIdx) {
//  	// silly, must pass instance since can't access outer class fields otherwise from a local inner class
//  	Instance fullInstance = instances.instance(instIdx); 
//  	int classValue = (int) fullInstance.classValue();
//  	if (classValue == m_majorityClasses[centroidIdx]) {
//  	  return true;
//  	} else {
//  	  return false;
//  	}
//        }
//      }
//      MajorityChecker majChecker = new MajorityChecker(majorityClasses);
//      //end debugging variance
    
//      int violatedConstraints = 0; 

//      for (int instIdx=0; instIdx<m_Instances.numInstances(); instIdx++) {
//        int centroidIdx = m_ClusterAssignments[instIdx];
//        diffInstance = m_metrics[centroidIdx].createDiffInstance(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));

//        for (int attr=0; attr<numAttributes; attr++) {
//  	weights[centroidIdx][attr] += diffInstance.value(attr); // Mahalanobis components
//  	if (debugVariance && instIdx < m_TotalTrainWithLabels.numInstances()) {
//  	  if (majChecker.belongsToMajority(m_TotalTrainWithLabels, instIdx, centroidIdx)) {
//  	    trueWeights[centroidIdx][attr] += diffInstance.value(attr);
//  	  } 
//  	}
//        }
//        counts[centroidIdx]++;

//        Object list =  m_instanceConstraintHash.get(new Integer(instIdx));
//        if (list != null) {   // there are constraints associated with this instance
//  	ArrayList constraintList = (ArrayList) list;
//  	for (int i = 0; i < constraintList.size(); i++) {
//  	  InstancePair pair = (InstancePair) constraintList.get(i);
//  	  int firstIdx = pair.first;
//  	  int secondIdx = pair.second;
//  	  double cost = 0;
//  	  if (pair.linkType == InstancePair.MUST_LINK) {
//  	    cost = m_MLweight;
//  	  } else if (pair.linkType == InstancePair.CANNOT_LINK) {
//  	    cost = m_CLweight;
//  	  }

//  	  Instance instance1 = m_Instances.instance(firstIdx);
//  	  Instance instance2 = m_Instances.instance(secondIdx);
//  	  int otherIdx = (firstIdx == instIdx) ? m_ClusterAssignments[secondIdx] : m_ClusterAssignments[firstIdx];

//  	  // check whether the constraint is violated
//  	  if (otherIdx != -1) {  
//  	    if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) { // violated must-link
//  	      if (m_verbose) {
//  		System.out.println("Found violated must link between: " + firstIdx + " and " + secondIdx);
//  	      }

//  	      // we penalize weights for both clusters involved, splitting the penalty in half
//  	      Instance diffInstance1 = m_metrics[otherIdx].createDiffInstance(instance1, instance2);
//  	      Instance diffInstance2 = m_metrics[centroidIdx].createDiffInstance(instance1, instance2);
	      
//  	      for (int attr=0; attr<numAttributes; attr++) {  // double-counting constraints, hence 0.5*0.5
//  		weights[otherIdx][attr] += 0.25 * cost * diffInstance1.value(attr);
//  		weights[centroidIdx][attr] += 0.25 * cost * diffInstance2.value(attr);
//  	      }
//  	      violatedConstraints++; 
//  	    }
	      
//  	    else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) { //violated cannot-link
//  	      if (m_verbose) {
//  		System.out.println("Found violated cannot link between: " + firstIdx + " and " + secondIdx);
//  	      }

//  	      // we penalize weights for just one cluster involved
//  	      diffInstance = m_metrics[centroidIdx].createDiffInstance(instance1, instance2);
//  	      Instance cannotDiffInstance = m_metrics[otherIdx].createDiffInstance(m_maxCLPoints[centroidIdx][0],
//  										   m_maxCLPoints[centroidIdx][1]);
//  	      for (int attr=0; attr<numAttributes; attr++) {  // double-counting constraints, hence 0.5
//  		weights[centroidIdx][attr] += 0.5 * cost * cannotDiffInstance.value(attr);
//  		weights[centroidIdx][attr] -= 0.5 * cost * diffInstance.value(attr); 
//  	      }
//  	      violatedConstraints++; 
//  	    }
//  	  } // end while
//  	}
//        }
//      }
//      System.out.println("   Total constraints violated: " + violatedConstraints/2 + "; per-cluster weights are:");
    
//      // check if NR needed
//      double [][] newWeights = new double[m_NumClusters][numAttributes];
//      double [][] currentWeights = new double[m_NumClusters][numAttributes];
//      for (int i=0; i<m_NumClusters; i++) {
//        currentWeights[i] = ((LearnableMetric) m_metrics[i]).getWeights();
//      }

//      for (int i=0; i<m_NumClusters; i++) {
//        boolean needNewtonRaphson = false;
//        for (int attr=0; attr<numAttributes; attr++) {
//  	if (weights[i][attr] < 0) { // check to avoid divide by 0
//  	  System.out.println("WARNING!  Cluster " + i + ", attribute " + attr + " weight=" + weights[i][attr]);


//  	  Cluster currentCluster = (Cluster) getClusters().get(i);
//  	  System.out.println("\nCluster " + i + ": " + currentCluster.size() + " instances");
//  	  if (currentCluster == null) {
//  	    System.out.println("(empty)");
//  	  }
//  	  else {
//  	    for (int j=0; j<currentCluster.size(); j++) {
//  	      Instance instance = (Instance) currentCluster.get(j);	
//  	      System.out.println("Instance: " + instance);
//  	    }
//  	  }
	  
//  	  needNewtonRaphson = true;
//  	  break;
//  	} else if (weights[i][attr] == 0) {
//  	  newWeights[i][attr] = currentWeights[i][attr];
//  	  System.out.println("WARNING!  Cluster " + i + ", attribute " + attr + " has 0 weight; keeping it as " + weights[i][attr]);
//  	} else {
//  	  newWeights[i][attr] = m_logTermWeight * counts[i]/weights[i][attr]; // invert weights
//  	  if (debugVariance) {
//  	    trueWeights[i][attr] = counts[i]/trueWeights[i][attr];
//  	  }
//  	}
//        }
      
//        // uncomment next line for debugging NR
//        // needNewtonRaphson = true;

//        // do NR if needed
//        if (needNewtonRaphson) {
//  	// weights not inverted here -- done in NR routine
//  	newWeights[i] = updateWeightsUsingNewtonRaphson(currentWeights[i], weights[i]);
	
//  	System.out.println(" (NR) ");
//        } 

//        // PRINT routine
//        //        System.out.print("\t" + i + "(" + counts[i] + "): ");
//        //        for (int attr=0; attr<numAttributes; attr++) {
//        //  	if (debugVariance) {
//        //  	  System.out.print(((float)trueWeights[i][attr]) + "/~/");
//        //  	} 
//        //  	System.out.print(((float)newWeights[i][attr]) + "\t");
//        //        }
//        //        System.out.println();
//        //        System.out.println("\t\tMean: " + m_ClusterCentroids.instance(i));
//        // end PRINT routine

//        ((LearnableMetric) m_metrics[i]).setWeights(newWeights[i]);
//      }
//      return true;
//    }



  /**
   * Gets the current settings of KL
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [1];

    int current = 0;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  public void setOptions(String[] options) throws Exception {
    // TODO: add later 
  }

  public Enumeration listOptions() {
    // TODO: add later 
    return null;
  }
}
