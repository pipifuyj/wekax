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
 *    MahalanobisLearner.java
 *    Copyright (C) 2004 Mikhail Bilenko and Sugato Basu
 *
 */

package weka.clusterers.metriclearners; 

import java.util.*;

import weka.core.*;
import weka.core.metrics.*;
import weka.clusterers.MPCKMeans;
import weka.clusterers.InstancePair;

import Jama.Matrix; 


/** 
 * A closed-form based learner for Mahalanobis
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu)
 * @version $Revision: 1.5 $ */

public class MahalanobisLearner extends MPCKMeansMetricLearner {
  /** min difference of objective function values for convergence*/
  protected double m_minDet = 1e-5;


  public void resetLearner() {
  } 

  /** if clusterIdx is -1, all instances are used
   * (a single metric for all clusters is used) */   
  public boolean trainMetric(int clusterIdx) throws Exception {
    Init(clusterIdx);

    Matrix updateMatrix = new Matrix(m_numAttributes, m_numAttributes);
    int violatedConstraints = 0;
    int numInstances = 0;

    WeightedMahalanobis metric = (WeightedMahalanobis) m_metric;
    Matrix maxMatrix = null;
    if (m_instanceConstraintMap.size() > 0) {
      if (clusterIdx == -1) { 
	maxMatrix = metric.createDiffMatrix(m_kmeans.m_maxCLPoints[0][0],
					    m_kmeans.m_maxCLPoints[0][1]);
      } else {
	maxMatrix = metric.createDiffMatrix(m_kmeans.m_maxCLPoints[clusterIdx][0],
					    m_kmeans.m_maxCLPoints[clusterIdx][1]);
      } 
      maxMatrix = maxMatrix.times(0.5);
    }

    for (int instIdx = 0; instIdx < m_instances.numInstances(); instIdx++) {
      int assignment = m_clusterAssignments[instIdx];

      // only instances assigned to this cluster are of importance
      if (assignment == clusterIdx || clusterIdx == -1) {
	numInstances++;
	if (clusterIdx < 0) {
	  m_centroid = m_kmeans.getClusterCentroids().instance(assignment); 
	}

	Instance instance = m_instances.instance(instIdx); 
	Matrix diffMatrix = metric.createDiffMatrix(instance, m_centroid); 
	updateMatrix = updateMatrix.plus(diffMatrix);

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
	    if (otherIdx != -1 ) {  
	      if (otherIdx != assignment && linkType == InstancePair.MUST_LINK) {
		diffMatrix = metric.createDiffMatrix(instance1, instance2);
		diffMatrix = diffMatrix.times(0.5);
		updateMatrix = updateMatrix.plus(diffMatrix); 
		violatedConstraints++; 
	      } else if (otherIdx == assignment && linkType == InstancePair.CANNOT_LINK) {
		diffMatrix = metric.createDiffMatrix(instance1, instance2);
		diffMatrix = diffMatrix.times(0.5);
		updateMatrix = updateMatrix.plus(maxMatrix); 
		updateMatrix = updateMatrix.minus(diffMatrix);
		violatedConstraints++; 
	      }
	    } // end while
	  }
	}
      }
    }
    updateMatrix = updateMatrix.times(1.0/numInstances);
    double updateDet = updateMatrix.det();
    int maxIterations = 1000;
    int currIteration = 1;
    Matrix newWeights = null;

//      System.out.println("UPDATE weights: " + " (violated constraints: " + violatedConstraints + ")");
//      for (int i = 0; i < updateMatrix.getArray().length; i++) {
//        for (int j = 0; j < updateMatrix.getArray()[i].length; j++) {
//    	System.out.print((float)updateMatrix.getArray()[i][j] + "\t");
//        }
//        System.out.println();
//      }

    // check that the update matrix is non-singular
    while (Math.abs(updateDet) < m_minDet && currIteration++ < maxIterations) {
      Matrix regularizer = Matrix.identity(m_numAttributes, m_numAttributes);
      regularizer = regularizer.times(updateMatrix.trace() * 0.01);
      updateMatrix = updateMatrix.plus(regularizer);
      System.out.print("\t" + currIteration + ". Singular update matrix, DET=" + (float)updateDet);
      updateDet = updateMatrix.det();
      System.out.println("; after regularization DET=" + (float)updateDet);
    }
    
    if (currIteration >= maxIterations) {      // if the matrix is irrepairable, return to identity matrix
      System.out.println("\n\nCOULDN'T REGULARIZE; GOING TO IDENTITY\n\n");
      newWeights = Matrix.identity(m_numAttributes, m_numAttributes);
    } else { 
      newWeights = updateMatrix.inverse();
    } 

    //      // check that matrix is positive semi-definite
    //      currIteration = 0;
    //      double det = newWeights.det();
    //      Matrix weightsSquare = newWeights.chol().getL();
    //      double sqDet = weightsSquare.det();
    //      while ((det < 0 || Math.abs(det) < m_ObjFunConvergenceDifference
    //  	    || Math.abs(sqDet) < m_ObjFunConvergenceDifference || Double.isNaN(sqDet))
    //  	   && currIteration++ < maxIterations) {
    //        // make sure the the matrix is symmetric positive definite
    //        if (det < 0) {
    //  	EigenvalueDecomposition ed = newWeights.eig();
    //  	Matrix eigenVectorsMatrix = ed.getV();
    //  	double[] evalues = ed.getRealEigenvalues();
    //  	double [][] evaluesM = new double[evalues.length][evalues.length];
    //  	for (int i = 0; i < evalues.length; i++) {
    //  	  if (evalues[i] < 0) {
    //  	    evalues[i] = -evalues[i];
    //  	  } else {
    //  	    evaluesM[i][i] = evalues[i];
    //  	  }
    //  	}
    //  	Matrix eigenValuesMatrix = new Matrix(evaluesM); 
    //  	// update the weights:  A' = V' * E * V
    //  	newWeights = ((eigenVectorsMatrix.transpose()).times(eigenValuesMatrix)).times(eigenVectorsMatrix);
    //  	System.out.println("\tNegative determinant; projecting for subsequent regularization");
    //        }

    //        // the weights matrix may end up singular (if determinant was negative, or det(updateMatrix) was very large
    //        sqDet = newWeights.chol().getL().det();
    //        det = newWeights.det();
    //        if (Math.abs(det) < m_ObjFunConvergenceDifference || Math.abs(sqDet) < m_ObjFunConvergenceDifference
    //  	  || Double.isNaN(sqDet)) {
    //  	Matrix regularizer = Matrix.identity(m_numAttributes, m_numAttributes);
    //  	regularizer = regularizer.times(newWeights.trace() * 0.01);
    //  	newWeights = newWeights.plus(regularizer);  // W = W + 0.01tr(W) * I
    //  	System.out.println("\tsingular matrix, det=" + ((float)det) + ", sqDet=" + ((float)sqDet) +
    //     "\tafter FIXING AND REGULARIZATION det=" + newWeights.det());
    //  	det = newWeights.det();
    //  	sqDet = newWeights.chol().getL().det();
    //        }
    //      }
    //      // if the matrix is irrepairable, return to identity matrix
    //      if (currIteration >= maxIterations) { 
    //        newWeights = Matrix.identity(m_numAttributes, m_numAttributes);
    //      }

    
    metric.setWeights(newWeights);

    // project all the instances for subsequent calculation of max-points for cannot-link penalties
    for (int instIdx=0; instIdx<m_instances.numInstances(); instIdx++) {
      if (clusterIdx < 0 || m_clusterAssignments[instIdx] == clusterIdx) { 
	metric.projectInstance(m_instances.instance(instIdx));
      }
    }
    return true; 

  }

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



//    protected void updateMetricWeightsMahalanobisGD() throws Exception {
//      WeightedMahalanobis metric = (WeightedMahalanobis) m_metric;
		
//      int numAttributes = m_Instances.numAttributes();
//      Instance diffInstance;
//      int violatedConstraints = 0;
//      Matrix newWeights = metric.getWeightsMatrix().copy();

//      // Do the GD
//      int iteration = 0;
//      boolean converged = false;

//      // precompute the update matrix for maxCannotLinkInstance
//      double[][] maxCLUpdate = new double[numAttributes][numAttributes];
//      Instance maxCLDiffInstance = null; 
//      if (m_maxCLPoints != null) { 
//        maxCLDiffInstance = metric.createDiffInstance(m_maxCLPoints[0][0],
//  						    m_maxCLPoints[0][1]);

//        for (int i = 0; i < numAttributes; i++) {
//  	for (int j = 0; j <=i; j++) {
//  	  maxCLUpdate[i][j] =
//  	    maxCLUpdate[j][i] =
//  	    maxCLDiffInstance.value(i) *maxCLDiffInstance.value(j);
//  	}
//        }
//      }
		
//      // store the constant part of the gradient:
//      double[][] gradientConst = new double[numAttributes][numAttributes];

//      for (int instIdx = 0; instIdx < m_Instances.numInstances(); instIdx++) {

//        // the (x-m)(x-m)' part
//        int centroidIdx = m_ClusterAssignments[instIdx];
//        Instance centroid = m_ClusterCentroids.instance(centroidIdx);
//        diffInstance = metric.createDiffInstance(m_Instances.instance(instIdx),
//  					       centroid);
//        for (int i = 0; i < numAttributes; i++) {
//  	for (int j = 0; j <= i; j++) {
//  	  gradientConst[i][j] =
//  	    gradientConst[j][i] =	diffInstance.value(i) * diffInstance.value(j); 
//  	}
//        }

//        // the violated constraints
//        Object list =  m_instanceConstraintHash.get(new Integer(instIdx));
//        if (list != null) {   // there are constraints associated with this instance
//  	ArrayList constraintList = (ArrayList) list;
//  	for (int constrIdx = 0; constrIdx < constraintList.size(); constrIdx++) {
//  	  InstancePair pair = (InstancePair) constraintList.get(constrIdx);
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
//  	  int otherIdx = (firstIdx == instIdx) ? m_ClusterAssignments[secondIdx]
//  	    : m_ClusterAssignments[firstIdx];
//  	  if (otherIdx == -1) {
//  	    throw new Exception("One of the instances is unassigned in "
//  				+ "updateMetricWeightsMahalanobisGD"); 
//  	  }

//  	  // check whether the constraint is violated
//  	  if (otherIdx != centroidIdx &&
//  	      pair.linkType == InstancePair.MUST_LINK) {
//  	    diffInstance = metric.createDiffInstance(instance1, instance2);
//  	    for (int i = 0; i < numAttributes; i++) {
//  	      for (int j = 0; j <= i; j++) {
//  		gradientConst[i][j] =
//  		  gradientConst[j][i] =
//  		  0.5 * cost * diffInstance.value(i) * diffInstance.value(j);
//  	      }
//  	    }
//  	    violatedConstraints++; 

//  	  } else if (otherIdx == centroidIdx &&
//  		     pair.linkType == InstancePair.CANNOT_LINK) {
//  	    diffInstance = metric.createDiffInstance(instance1, instance2);
//  	    for (int i = 0; i < numAttributes; i++) {
//  	      for (int j = 0; j <= i; j++) {
//  		gradientConst[i][j] =
//  		  gradientConst[j][i] =
//  		  0.5 * cost *
//  		  (maxCLUpdate[i][j] -
//  		   diffInstance.value(i) * diffInstance.value(j)); 
//  	      }
//  	    }
//  	    violatedConstraints++; 
//  	  }
//  	}
//        }
//      }
//      Matrix constUpdate = new Matrix(gradientConst); 
		

//      while (iteration < m_maxGDIterations && !converged) {
//        // calculate the gradient
//        Matrix update =  constUpdate.copy(); 

//        // factor in the A^-1 
//        Matrix Ai = newWeights.inverse();
//        Ai.timesEquals(m_logTermWeight); 
//        update.minusEquals(Ai); 

//        // regularization  (-1/sum(a_ij)^2)
//        double regularizer = 0; 
//        for (int i = 0; i < numAttributes; i++) {
//  	for (int j = 0; j <= i; j++) {
//  	  regularizer += 2.0/(newWeights.get(i, j) * newWeights.get(i, j));
//  	}
//        }
//        // correct for double-counted diagonal
//        for (int i = 0; i < numAttributes; i++) {
//  	regularizer -= 1.0/newWeights.get(i, i);
//        }

//        regularizer *= m_currregularizerTermWeight; 
//        for (int i = 0; i < numAttributes; i++) {
//  	for (int j = 0; j < numAttributes; j++) {
//  	  update.set(i, j, update.get(i,j) - regularizer);
//  	}
//        }

//        // update
//        update.timesEquals(m_currEta); 
//        newWeights.minusEquals(update);

//        // anneal if necessary and check for convergence
//        m_currEta = m_currEta * m_etaDecayRate;

//        // check for convergence
//        double norm = update.norm1();
//        System.out.println(iteration + ":  norm=" + norm); 
//        if (norm < 0.0001) {
//  	converged = true;
//        }
//        iteration++; 
//      }

//      // We're done, set the weights to newWeights
		
//    }



// MULTIPLE:




//    /** M-step of the KMeans clustering algorithm -- updates metric
//     *  weights. Invoked only when metric is an instance of Mahalanobis
//     * @return value true if everything was alright; false if there was
//     miserable failure and clustering needs to be restarted */
//    protected boolean updateMultipleMetricWeightsMahalanobis() throws Exception {
//      if (m_regularizeWeights) {
//        System.out.println("Regularized version, calling GD version of updateMultipleMetricWeightsMahalanobisGD!");
//        updateMultipleMetricWeightsMahalanobisGD();
//      }

//      int numAttributes = m_Instances.numAttributes();
//      if (m_Instances.classIndex() >= 0) {
//        numAttributes--;
//      }

//      Matrix [] updateMatrices = new Matrix[m_metrics.length];
//      for (int i = 0; i < updateMatrices.length; i++) { 
//        updateMatrices[i] = new Matrix(numAttributes, numAttributes);
//      }
//      int violatedConstraints = 0;
//      int [] counts = new int[updateMatrices.length];

//      for (int instIdx=0; instIdx<m_Instances.numInstances(); instIdx++) {
//        int centroidIdx = m_ClusterAssignments[instIdx];
//        Matrix diffMatrix = ((WeightedMahalanobis) m_metrics[centroidIdx]).createDiffMatrix(m_Instances.instance(instIdx),
//  											  m_ClusterCentroids.instance(centroidIdx));
//        updateMatrices[centroidIdx] = updateMatrices[centroidIdx].plus(diffMatrix);
//        counts[centroidIdx]++;

//        // go through violated constraints
//        Object list =  m_instanceConstraintHash.get(new Integer(instIdx));
//        if (list != null) {   // there are constraints associated with this instance
//  	ArrayList constraintList = (ArrayList) list;
//  	for (int i = 0; i < constraintList.size(); i++) {
//  	  InstancePair pair = (InstancePair) constraintList.get(i);
//  	  int firstIdx = pair.first;
//  	  int secondIdx = pair.second;
//  	  Instance instance1 = m_Instances.instance(firstIdx);
//  	  Instance instance2 = m_Instances.instance(secondIdx);
//  	  int otherIdx = (firstIdx == instIdx) ? m_ClusterAssignments[secondIdx] : m_ClusterAssignments[firstIdx];

//  	  // check whether the constraint is violated
//  	  if (otherIdx != -1) {  
//  	    if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) {
//  	      Matrix diffMatrix1 = ((WeightedMahalanobis) m_metrics[centroidIdx]).createDiffMatrix(instance1, instance2);
//  	      diffMatrix1 = diffMatrix1.times(0.25);
//  	      Matrix diffMatrix2 = ((WeightedMahalanobis) m_metrics[otherIdx]).createDiffMatrix(instance1, instance2);
//  	      diffMatrix2 = diffMatrix2.times(0.25);
//  	      updateMatrices[centroidIdx] = updateMatrices[centroidIdx].plus(diffMatrix1); 
//  	      updateMatrices[otherIdx] = updateMatrices[otherIdx].plus(diffMatrix2);
//  	      violatedConstraints++; 
//  	    } else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) {
//  	      diffMatrix = ((WeightedMahalanobis) m_metrics[centroidIdx]).createDiffMatrix(instance1, instance2);
//  	      Matrix maxMatrix = ((WeightedMahalanobis) m_metrics[centroidIdx]).createDiffMatrix(m_maxCLPoints[centroidIdx][0],
//  												 m_maxCLPoints[centroidIdx][1]);
//  	      diffMatrix = diffMatrix.times(0.5);
//  	      maxMatrix = maxMatrix.times(0.5);
//  	      updateMatrices[centroidIdx] = updateMatrices[centroidIdx].plus(maxMatrix); 
//  	      updateMatrices[centroidIdx] = updateMatrices[centroidIdx].minus(diffMatrix);
//  	      violatedConstraints++; 
//  	    }
//  	  } // end while
//  	}
//        }
//      }
   
//      int [][] classCounts  = new int[m_NumClusters][m_TotalTrainWithLabels.numClasses()];
//      // NB:  m_TotalTrainWithLabels does *not* include unlabeled data, counts here are undersampled!
//      // assuming unlabeled data came from same distribution as m_TotalTrainWithLabels, counts are still valid...
//      for (int instIdx=0; instIdx<m_TotalTrainWithLabels.numInstances(); instIdx++) {
//        Instance fullInstance = m_TotalTrainWithLabels.instance(instIdx);
//        classCounts[m_ClusterAssignments[instIdx]][(int)(fullInstance.classValue())]++;
//      }
//      for (int i = 0; i < m_NumClusters; i++){
//        System.out.print("Cluster " + i + "(" + counts[i] + ")\t" + classCounts[i][0]);
//        for (int j = 1; j < m_TotalTrainWithLabels.numClasses(); j++) {
//  	System.out.print("\t" + classCounts[i][j]);
//        }
//        System.out.println();
//      }

//      // now update the actual weight matrices
//      for (int i = 0; i < updateMatrices.length; i++) {
//        int maxIterations = 100;
//        if (counts[i] == 0) {
//  	//System.out.println("Cluster " + i + " has lost all instances; leaving weights as is");
//  	updateMatrices[i] = Matrix.identity(numAttributes, numAttributes);
//  	counts[i] = 1;
//  	//System.err.println("IRREPAIRABLE COVARIANCE MATRIX, RESTARTING");
//  	//return false;
//        }
//        updateMatrices[i] = updateMatrices[i].times(1.0/counts[i]);
//        double updateDet = updateMatrices[i].det();
//        int currIteration = 0;
//        Matrix newWeights = null; 

//        // check that the update matrix is non-singular
//        while (Math.abs(updateDet) < m_NRConvergenceDifference && currIteration++ < maxIterations) {
//  	Matrix regularizer = Matrix.identity(numAttributes, numAttributes);
//  	regularizer = regularizer.times(updateMatrices[i].trace() * 0.01);
//  	updateMatrices[i] = updateMatrices[i].plus(regularizer);
//  	System.out.print(i + "\tsingular UPDATE matrix, DET=" + ((float)updateDet));
//  	updateDet = updateMatrices[i].det();
//  	System.out.println("; after regularization DET=" + ((float)updateDet));
//  	//  	System.out.println("ACTUAL weights: ");
//  	//  	double[][] m_weights = updateMatrices[i].getArray();
//  	//  	for (int l = 0; l < m_weights.length; l++) {
//  	//  	  for (int j = 0; j < m_weights[l].length; j++) {
//  	//  	    System.out.print(((float)m_weights[l][j]) + "\t");
//  	//  	}
//  	//  	  System.out.println();
//  	//  	}
//        }
//        if (currIteration >= maxIterations) {      // if the matrix is irrepairable, return to identity matrix
//  	newWeights = Matrix.identity(numAttributes, numAttributes);
//  	System.err.println("IRREPAIRABLE UPDATE MATRIX, RESTARTING");
//        } else { 
//  	newWeights = updateMatrices[i].inverse();
//        } 
//        ((WeightedMahalanobis) m_metrics[i]).setWeights(newWeights);

//        // project all the instances for subsequent calculation of max-points for cannot-link penalties
//        // TODO:  we are projecting ALL instances just in case...  possibly can optimize in the future
//        for (int instIdx=0; instIdx<m_Instances.numInstances(); instIdx++) {
//  	((WeightedMahalanobis) m_metrics[i]).projectInstance(m_Instances.instance(instIdx));
//        }
//      }

//      return true; 
//    }

