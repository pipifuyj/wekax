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
 *    KLLearner.java
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
 * A gradient-descent based learner for KL
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu)
 * @version $Revision: 1.8 $ */

public class KLGDLearner extends GDMetricLearner {

  /** if clusterIdx is -1, all instances are used
   * (a single metric for all clusters is used) */   
  public boolean trainMetric(int clusterIdx) throws Exception {
    Init(clusterIdx);

    if (((KL)m_metric).getUseIDivergence() == false) {
      System.out.println("Trainable KL metric, using IDvergence ...");
      ((KL)m_metric).setUseIDivergence(true);
    }

    int numInstances = 0; 
    double [] gradients = new double[m_numAttributes];
    Instance diffInstance;
    int violatedConstraints = 0;
    double [] currentWeights = m_metric.getWeights();
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

	// variance components
	if (diffInstance instanceof SparseInstance) {
	  for (int i = 0; i < diffInstance.numValues(); i++) {
	    int idx = diffInstance.index(i);
	    gradients[idx] += diffInstance.valueSparse(i); 
	  } 
	} else {  // non-sparse case
	  for (int attr=0; attr<m_numAttributes; attr++) {
	    gradients[attr] += diffInstance.value(attr); // variance components
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
	    int otherIdx = (firstIdx == instIdx) ? m_clusterAssignments[secondIdx]
	      : m_clusterAssignments[firstIdx];

	    // check whether the constraint is violated
	    if (otherIdx != -1) {  
	      if (otherIdx != assignment && linkType == InstancePair.MUST_LINK) {
		diffInstance = ((KL) m_metric).createDiffInstanceJS(instance1, instance2);
		if (diffInstance instanceof SparseInstance) {
		  for (int l = 0; l < diffInstance.numValues(); l++) {
		    int idx = diffInstance.index(l);
		    gradients[idx] += 0.5 * m_MLweight * diffInstance.valueSparse(l); 
		  } 
		} else {  // non-sparse case
		  for (int attr = 0; attr < m_numAttributes; attr++) {
		    gradients[attr] += 0.5 * m_MLweight * diffInstance.value(attr); // variance components
		  }
		}
		violatedConstraints++; 
	      } else if (otherIdx == assignment && linkType == InstancePair.CANNOT_LINK){
		diffInstance = ((KL) m_metric).createDiffInstanceJS(instance1, instance2);
		// Cannot link component, adjusted not to double count constraints
		if (diffInstance instanceof SparseInstance) {
		  for (int l = 0; l < diffInstance.numValues(); l++) {
		    int idx = diffInstance.index(l);
		    gradients[idx] -= 0.5 * m_CLweight * diffInstance.valueSparse(l); 
		  } 
		} else {  // non-sparse case
		  for (int attr=0; attr<m_numAttributes; attr++) {
		    gradients[attr] -= 0.5 * m_CLweight * diffInstance.value(attr); // variance components
		  }
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

  /**
   * Gets the current settings of KL
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [20];
    
    int current = 0;

    options[current++] = "-E";
    options[current++] = "" + m_eta;
    options[current++] = "-D";
    options[current++] = "" + m_etaDecayRate;

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




//    /** M-step of the KMeans clustering algorithm -- updates KL metric
//     *  weights for the individual metrics. Invoked only when metric is
//     *  trainable */
//    protected boolean updateMultipleMetricWeightsKLGD() throws Exception {
//      int numAttributes = m_Instances.numAttributes();
//      double [][] gradients = new double[m_metrics.length][numAttributes];
//      double [][] regularizerComponents = new double[m_metrics.length][numAttributes];
//      int [] clusterCounts = new int[m_NumClusters];
//      Instance diffInstance1, diffInstance2;
//      int violatedConstraints = 0; 
//      double [][] currentWeights = new double[m_metrics.length][numAttributes];
//      for (int i = 0; i < m_metrics.length; i++) { 
//        currentWeights[i] = ((LearnableMetric) m_metrics[i]).getWeights();
//      }

//      for (int i=0; i<m_NumClusters; i++){
//        for (int attr=0; attr<numAttributes; attr++) {
//  	regularizerComponents[i][attr] = 0;
//        }
//      }    

//      for (int instIdx=0; instIdx<m_Instances.numInstances(); instIdx++) {
//        int centroidIdx = m_ClusterAssignments[instIdx];
//        clusterCounts[centroidIdx]++;
//        diffInstance1 = ((LearnableMetric) m_metrics[centroidIdx]).createDiffInstance(m_Instances.instance(instIdx),
//  										    m_ClusterCentroids.instance(centroidIdx));
//        // Mahalanobis components 
//        for (int attr=0; attr<numAttributes; attr++) {
//  	gradients[centroidIdx][attr] += diffInstance1.value(attr); // variance components
//  	// SUGATO: Added regularization code
//  	if (currentWeights[centroidIdx][attr] > 0) {
//  	  if (m_m_regularizeWeights) {
//  	    regularizerComponents[centroidIdx][attr] += m_currm_regularizerTermWeight/(currentWeights[centroidIdx][attr] * currentWeights[centroidIdx][attr]); // regularization
//  	  } else {
//  	    regularizerComponents[centroidIdx][attr] = 0;
//  	  }
//  	}
//        }

//        // go through violated constraints
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
//  	    if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) {
//  	      // penalize both clusters involved
//  	      diffInstance1 = ((KL) m_metrics[otherIdx]).createDiffInstanceJS(instance1, instance2);
//  	      diffInstance2 = ((KL) m_metrics[centroidIdx]).createDiffInstanceJS(instance1, instance2);
//  	      for (int attr=0; attr<numAttributes; attr++) {
//  		gradients[otherIdx][attr] += 0.25 * cost * diffInstance1.value(attr);
//  		gradients[centroidIdx][attr] += 0.25 * cost * diffInstance2.value(attr);
//  	      }
//  	      violatedConstraints++; 
//  	    } else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) {
//  	      diffInstance1 = ((KL) m_metrics[otherIdx]).createDiffInstanceJS(instance1, instance2);

//  	      // Cannot link component, adjusted not to double count constraints
//  	      for (int attr=0; attr<numAttributes; attr++) {
//  		gradients[otherIdx][attr] -= 0.5 * cost * diffInstance1.value(attr); 
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
