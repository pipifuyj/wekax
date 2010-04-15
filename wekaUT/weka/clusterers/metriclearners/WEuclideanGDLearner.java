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
 *    WEuclideanLearnerGD.java
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
 * A gradient-descent based learner for WeightedEuclidean
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu)
 * @version $Revision: 1.5 $ */

public class WEuclideanGDLearner extends GDMetricLearner {


  /** if clusterIdx is -1, all instances are used
   * (a single metric for all clusters is used) */   
  public boolean trainMetric(int clusterIdx) throws Exception {
    Init(clusterIdx);

    double [] gradients = new double[m_numAttributes];
    Instance diffInstance;
    int violatedConstraints = 0; 
    double [] currentWeights = m_metric.getWeights();
    int numInstances = m_instances.numInstances();
    double [] regularizerComponents = new double[m_numAttributes];

    if (m_regularize) {
      regularizerComponents = InitRegularizerComponents(currentWeights); 
    }
    
    for (int instIdx = 0; instIdx < m_instances.numInstances(); instIdx++) {
      int assignment = m_clusterAssignments[instIdx];

      // only instances assigned to this cluster are of importance
      if (assignment == clusterIdx || clusterIdx == -1) {
	Instance instance = m_instances.instance(instIdx); 
	numInstances++;

	if (clusterIdx < 0) {
	  m_centroid = m_kmeans.getClusterCentroids().instance(assignment); 
	}
	
	diffInstance = m_metric.createDiffInstance(instance, m_centroid);
	for (int attr = 0; attr < m_numAttributes; attr++) {
	  gradients[attr] +=  diffInstance.value(attr); // Euclidean components
	  if (currentWeights[attr] > 0) {
	    gradients[attr] -= m_logTermWeight/currentWeights[attr]; // log components

//  	    if (m_regularize) {
//  	      regularizerComponents[attr] = m_regularizerTermWeight *
//  		m_metric.getRegularizer().gradient(currentWeights[attr]);
//  	    } else {
//  	      regularizerComponents[attr] = 0;
//  	    }
	  }
	}

	// go through violated constraints
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
	    int otherIdx = (firstIdx == instIdx) ?
	      m_clusterAssignments[secondIdx] : m_clusterAssignments[firstIdx];

	    // check whether the constraint is violated
	    if (otherIdx != -1) {  
	      if (otherIdx != assignment && linkType == InstancePair.MUST_LINK) {
		diffInstance = m_metric.createDiffInstance(instance1, instance2);
		for (int attr = 0; attr < m_numAttributes; attr++) {
		  gradients[attr] += 0.5 * m_MLweight * diffInstance.value(attr);
		}
		violatedConstraints++; 
	      } else if (otherIdx == assignment && linkType == InstancePair.CANNOT_LINK){
		diffInstance = m_metric.createDiffInstance(instance1, instance2);

		for (int attr = 0; attr < m_numAttributes; attr++) {
		  // this constraint will be counted twice, hence 0.5
		  gradients[attr] += 0.5 * m_CLweight * m_maxCLDiffInstance.value(attr);
		  gradients[attr] -= 0.5 * m_CLweight * diffInstance.value(attr); 
		}
		violatedConstraints++; 
	      }
	    } // end while
	  }
	}
      }
    }

    double [] newWeights = GDUpdate(currentWeights, gradients, regularizerComponents); 
    m_metric.setWeights(newWeights);
    
    System.out.println("   Total constraints violated: " + violatedConstraints/2); 
    return true; 
    
  }

/// OLD CODE FOR MULTIPLE

//    /** M-step of the KMeans clustering algorithm -- updates Euclidean
//     *  metric weights for the individual metrics using gradient
//     *  descent. Invoked only when m_regularizeWeights is true and
//     *  metric is trainable */
//    protected boolean updateMultipleMetricWeightsEuclideanGD() throws Exception {
//      // SUGATO: Added regularization code to updateMultipleMetricWeightsEuclideanGD
//      int numAttributes = m_Instances.numAttributes();
//      double[][] gradients = new double[m_NumClusters][numAttributes];
//      double[][] regularizerComponents = new double[m_NumClusters][numAttributes];
//      int [] clusterCounts = new int[m_NumClusters]; // count how many instances are in each cluster
//      Instance diffInstance;

//      double [][] currentWeights = new double[m_NumClusters][numAttributes];
//      for (int i=0; i<m_NumClusters; i++) {
//        currentWeights[i] = ((LearnableMetric) m_metrics[i]).getWeights();
//      }

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

//      for (int i=0; i<m_NumClusters; i++){
//        for (int attr=0; attr<numAttributes; attr++) {
//  	regularizerComponents[i][attr] = 0;
//        }
//      }

//      for (int instIdx=0; instIdx<m_Instances.numInstances(); instIdx++) {
//        int centroidIdx = m_ClusterAssignments[instIdx];
//        diffInstance = m_metrics[centroidIdx].createDiffInstance(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));

//        for (int attr=0; attr<numAttributes; attr++) {
//  	gradients[centroidIdx][attr] += diffInstance.value(attr); // Mahalanobis components
//  	if (currentWeights[centroidIdx][attr] > 0) {
//  	  gradients[centroidIdx][attr] -= 1/currentWeights[centroidIdx][attr]; // log components
//  	  if (m_regularizeWeights) {
//  	    regularizerComponents[centroidIdx][attr] += m_currregularizerTermWeight/(currentWeights[centroidIdx][attr] * currentWeights[centroidIdx][attr]);
//  	  } else {
//  	    regularizerComponents[centroidIdx][attr] = 0;
//  	  }
//  	}
//  	if (debugVariance && instIdx < m_TotalTrainWithLabels.numInstances()) {
//  	  if (majChecker.belongsToMajority(m_TotalTrainWithLabels, instIdx, centroidIdx)) {
//  	    trueWeights[centroidIdx][attr] += diffInstance.value(attr);
//  	  } 
//  	}
//        }
//        clusterCounts[centroidIdx]++;

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
//  	      cost = m_CLweight;
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
//  		gradients[otherIdx][attr] += 0.25 * cost * diffInstance1.value(attr);
//  		gradients[centroidIdx][attr] += 0.25 * cost * diffInstance2.value(attr);
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
//  		gradients[centroidIdx][attr] += 0.5 * cost * cannotDiffInstance.value(attr);
//  		gradients[centroidIdx][attr] -= 0.5 * cost * diffInstance.value(attr); 
//  	      }
//  	      violatedConstraints++; 
//  	    }
//  	  } // end while
//  	}
//        }
//      }

//      double [][] newWeights = new double[m_metrics.length][numAttributes];
//      for (int i = 0; i < m_metrics.length; i++) { 
//        for (int attr=0; attr<numAttributes; attr++) {
//  	gradients[i][attr] *= m_currEta;
//  	if (gradients[i][attr] > regularizerComponents[i][attr]) { // to take into account the direction of the gradient descent update
//  	  newWeights[i][attr] = currentWeights[i][attr] - gradients[i][attr] + regularizerComponents[i][attr];
//  	} else {
//  	  newWeights[i][attr] = currentWeights[i][attr] + gradients[i][attr] - regularizerComponents[i][attr];
//  	}
//  	if (newWeights[i][attr] < 0) {
//  	  System.out.println("prevented negative weight " + newWeights[i][attr] + " for attribute " + m_Instances.attribute(attr).name()); 
//  	  newWeights[i][attr] = 0;
//  	} else if (newWeights[i][attr] == 0) {
//  	  System.out.println("zero weight for attribute " + m_Instances.attribute(attr).name()); 
//  	}
//        }

//        newWeights[i] = ClusterUtils.normalize(newWeights[i]);
//        ((LearnableMetric) m_metrics[i]).setWeights(newWeights[i]);

//        // PRINT top weights
//        System.out.println("Cluster " + i + " (" + clusterCounts[i] + ")"); 
//        TreeMap map = new TreeMap(Collections.reverseOrder());
//        for (int j = 0; j < newWeights[i].length; j++) {
//  	map.put(new Double(newWeights[i][j]), new Integer(j));
//        }
//        Iterator it = map.entrySet().iterator();
//        for (int j=0; j < 10 && it.hasNext(); j++) {
//  	Map.Entry entry = (Map.Entry) it.next();
//  	int idx = ((Integer)entry.getValue()).intValue();
//  	System.out.println("\t" + m_Instances.attribute(idx).name() + "\t" + newWeights[i][idx] 
//  			   + "\tgradient=" + gradients[i][idx] + "\tregularizer=" + regularizerComponents[i][idx]);
//        }
//        // end PRINT top weights
//      }
//      m_currEta = m_currEta * m_etaDecayRate; 
//      //    m_currregularizerTermWeight *= m_etaDecayRate;
    
//      // PRINT routine
//      //      System.out.println("Total constraints violated: " + violatedConstraints/2 + "; weights are:"); 
//      //      for (int attr=0; attr<numAttributes; attr++) {
//      //        System.out.print(newWeights[attr] + "\t");
//      //      }
//      //      System.out.println();
//      // end PRINT routine

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
