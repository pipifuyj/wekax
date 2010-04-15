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
 *    WeightedMahalanobis.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import weka.clusterers.InstancePair; 
import weka.core.*;
import java.util.*;

import Jama.Matrix;
import Jama.EigenvalueDecomposition;

/** 
 * WeightedMahalanobis class
 *
 * Implements a weighted Mahalanobis distance metric weighted by a full matrix of weights. 
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.10 $
 */

public class WeightedMahalanobis extends  LearnableMetric implements OptionHandler {
  /** The full matrix of attribute weights */
  protected double[][] m_weightsMatrix = null;
  /** weights^0.5, used to project instances to the new space to speed up calculations */
  protected double[][] m_weightsMatrixSquare = null; 

  /** A hash where instances are projected using the weights */
  protected HashMap m_projectedInstanceHash = null;

  /** Max instance storage (max is in the space of projected instances, values are in **ORIGINAL** space)
   *  Currently somewhat convoluted, TODO:  re-write the max value code in MPCKMeans */
  protected double [][] m_maxPoints = null;
  // store the hypercube in the projected space
  protected double [][] m_maxProjPoints = null;  

  /** We can have different ways of converting from distance to similarity  */
  public static final int CONVERSION_LAPLACIAN = 1;
  public static final int CONVERSION_UNIT = 2;
  public static final int CONVERSION_EXPONENTIAL = 4;
  public static final Tag[] TAGS_CONVERSION = {
    new Tag(CONVERSION_UNIT, "similarity = 1-distance"),
    new Tag(CONVERSION_LAPLACIAN, "similarity=1/(1+distance)"),
    new Tag(CONVERSION_EXPONENTIAL, "similarity=exp(-distance)")
      };
  /** The method of converting, by default laplacian */
  protected int m_conversionType = CONVERSION_LAPLACIAN;

  /**
   * Create a new metric.
   * @param numAttributes the number of attributes that the metric will work on
   */ 
  public WeightedMahalanobis(int numAttributes) throws Exception {
    buildMetric(numAttributes);
  }

  /** Create a default new metric */
  public WeightedMahalanobis() {
  } 
   
  /**
   * Creates a new metric which takes specified attributes.
   *
   * @param _attrIdxs An array containing attribute indeces that will
   * be used in the metric
   */
  public WeightedMahalanobis(int[] _attrIdxs) throws Exception {
    setAttrIdxs(_attrIdxs);
    buildMetric(_attrIdxs.length);	
  }

  /**
   * Reset all values that have been learned
   */
  public void resetMetric() throws Exception {
    m_trained = false;
    m_projectedInstanceHash = new HashMap();
    if (m_weightsMatrix != null) { 
      for (int i = 0; i < m_weightsMatrix.length; i++) {
	Arrays.fill(m_weightsMatrix[i], 0);
	Arrays.fill(m_weightsMatrixSquare[i], 0);
	m_weightsMatrix[i][i] = 1;
	m_weightsMatrixSquare[i][i] = 1; 
      }
    }
    recomputeNormalizer();
    recomputeRegularizer();
  }

  /**
   * Generates a new Metric. Has to initialize all fields of the metric
   * with default values.
   *
   * @param numAttributes the number of attributes that the metric will work on
   * @exception Exception if the distance metric has not been
   * generated successfully.
   */
  public void buildMetric(int numAttributes) throws Exception {
    m_numAttributes = numAttributes;
    m_weightsMatrix = new double[numAttributes][numAttributes];
    m_weightsMatrixSquare = new double[numAttributes][numAttributes];
    m_maxProjPoints = new double[2][numAttributes];
    m_attrIdxs = new int[numAttributes];
    for (int i = 0; i < numAttributes; i++) {
      m_attrIdxs[i] = i;
    }
    resetMetric();
  }

    
  /**
   * Generates a new Metric. Has to initialize all fields of the metric
   * with default values
   *
   * @param options an array of options suitable for passing to setOptions.
   * May be null. 
   * @exception Exception if the distance metric has not been
   * generated successfully.
   */
  public void buildMetric(int numAttributes, String[] options) throws Exception {
    buildMetric(numAttributes);
  }

  /**
   * Create a new metric for operating on specified instances
   * @param data instances that the metric will be used on
   */
  public  void buildMetric(Instances data) throws Exception {
    m_classIndex = data.classIndex();
    m_numAttributes = data.numAttributes();
    if (m_classIndex != m_numAttributes-1 && m_classIndex != -1) {
      throw new Exception("Class attribute (" + m_classIndex + ") should be the last attribute!!!");
    }
    if (m_classIndex != -1) {
      m_numAttributes--;
    }
    System.out.println("About to build metric with " + m_numAttributes + " attributes, trainable=" + m_trainable);
    buildMetric(m_numAttributes);
    if (m_trainable) {
      learnMetric(data);
    }
  }
    
  
  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distance(Instance instance1, Instance instance2) throws Exception {
    double distance = 0; 
    if (m_weightsMatrixSquare != null) {
      double [] projValues1;
      double [] projValues2;
      if (m_projectedInstanceHash.containsKey(instance1)) {
	projValues1 = (double []) m_projectedInstanceHash.get(instance1);
      } else {
	projValues1 = projectInstance(instance1); 
      }

      if (m_projectedInstanceHash.containsKey(instance2)) {
	projValues2 = (double []) m_projectedInstanceHash.get(instance2);
      } else {
	projValues2 = projectInstance(instance2); 
      }

      double diff = 0;
      for (int i = 0; i < projValues1.length; i++) {
	if (i != m_classIndex) {
	  diff = projValues1[i] - projValues2[i];
	  distance += diff * diff;
	}
      }
    } else {  // do the full matrix computation
      //      System.out.println("full matrix");
      double[] diffValues = new double[m_weightsMatrix.length];
      for (int i = 0; i < diffValues.length; i++) {
	diffValues[i] = instance1.value(i) - instance2.value(i);
      }

      double [] xyM = new double[m_weightsMatrix.length];
      for (int i = 0; i < m_weightsMatrix.length; i++) {
	for (int j = 0; j < m_weightsMatrix.length; j++) {
	  xyM[i] += diffValues[j] * m_weightsMatrix[j][i];
	}
      }

      for (int i = 0; i < m_weightsMatrix.length; i++) {
	distance += xyM[i] * diffValues[i];
      }
    }
    distance = Math.sqrt(distance);
    return distance;
  }

  /** Return the penalty contribution - distance*distance */
  public double penalty(Instance instance1,
			Instance instance2) throws Exception {
    double distance = distance(instance1, instance2);
    return distance * distance; 
  }

  /** Return the penalty contribution - distance*distance */
  public double penaltySymmetric(Instance instance1,
			Instance instance2) throws Exception {
    double distance = distance(instance1, instance2);
    return distance * distance; 
  }
  

  /** given an instance, project it using the weights matrix and store it in the hash */
  public double[] projectInstance(Instance instance) {
    int numValues = instance.numValues();
    double[] values = instance.toDoubleArray();
    double[] projValues = new double[numValues];

    if (m_weightsMatrixSquare == null) {
      return null; 
    } 

    if (m_maxProjPoints == null) { 
      m_maxPoints = new double [2][m_weightsMatrix.length];
      m_maxProjPoints = new double[2][m_weightsMatrix.length];
      Arrays.fill(m_maxProjPoints[0], Double.MAX_VALUE);
      Arrays.fill(m_maxProjPoints[1], Double.MIN_VALUE);
    }
    
    for (int i = 0; i < m_weightsMatrix.length; i++) {
      if (i != m_classIndex) {
	for (int j = 0; j < m_weightsMatrix.length; j++) {
	  projValues[i] += values[j] * m_weightsMatrixSquare[j][i];
	}

	// update the enclosing maxPoints
	if (projValues[i] < m_maxProjPoints[0][i]) {
	  m_maxProjPoints[0][i] = projValues[i];
	}
	if (projValues[i] > m_maxProjPoints[1][i]) {
	  m_maxProjPoints[1][i] = projValues[i];
	} 
      } else { // class attribute
	projValues[i] = values[i];
      } 
    }
    m_projectedInstanceHash.put(instance, projValues);
    return projValues;
  }

  /** Get the maxPoints instances */
  public double [][] getMaxPoints(HashMap constraintMap, Instances instances) throws Exception {
    m_maxPoints = new double [2][m_weightsMatrix.length];
    InstancePair maxConstraint = null;
    double maxDistance = -Double.MIN_VALUE; 
    Iterator iterator = constraintMap.entrySet().iterator();
    while (iterator.hasNext()) {
      Map.Entry entry = (Map.Entry) iterator.next();
      int type = ((Integer) entry.getValue()).intValue();
      if (type == InstancePair.CANNOT_LINK) {
	InstancePair pair = (InstancePair) entry.getKey();
	int firstIdx = pair.first;
	int secondIdx = pair.second;
	Instance instance1 = instances.instance(firstIdx);
	Instance instance2 = instances.instance(secondIdx);
	double distance = distance(instance1, instance2);
	if (distance >= maxDistance) {
	  maxConstraint = pair;
	  maxDistance = distance; 
	} 
      } 
    }
    if (maxDistance == -Double.MIN_VALUE) {
      //  System.out.println("ACTUAL weights det=" + ((new Matrix(m_weightsMatrix)).det()));
//        for (int i = 0; i < m_weightsMatrix.length; i++) {
//  	for (int j = 0; j < m_weightsMatrix[i].length; j++) {
//  	  System.out.print(((float)m_weightsMatrix[i][j]) + "\t");
//  	}
//  	System.out.println();
//        }
//        System.out.println("\n\nsq weights");
//        for (int i = 0; i < m_weightsMatrixSquare.length; i++) {
//  	for (int j = 0; j < m_weightsMatrixSquare[i].length; j++) {
//  	  System.out.print(((float)m_weightsMatrixSquare[i][j]) + "\t");
//  	}
//  	System.out.println();
//        }
      if (m_weightsMatrixSquare != null) { 
	m_weightsMatrixSquare = null;
	System.out.println("recursing");
	return getMaxPoints(constraintMap, instances);
      } else {
	iterator = constraintMap.entrySet().iterator();
	while (iterator.hasNext()) {
	  Map.Entry entry = (Map.Entry) iterator.next();
	  int type = ((Integer) entry.getValue()).intValue();
	  if (type == InstancePair.CANNOT_LINK) {
	    maxConstraint = (InstancePair) entry.getKey();
	    break;
	  }
	}
      } 
    } 
    int firstIdx = maxConstraint.first;
    int secondIdx = maxConstraint.second;
    Instance instance1 = instances.instance(firstIdx);
    Instance instance2 = instances.instance(secondIdx);
    for (int i = 0; i < m_weightsMatrix.length; i++) {
      m_maxPoints[0][i] = instance1.value(i);
      m_maxPoints[1][i] = instance2.value(i);
    }
    return m_maxPoints;
  } 

  /**
   * Returns a non-weighted distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceNonWeighted(Instance instance1, Instance instance2) throws Exception {
    double value1, value2, diff, distance = 0;
    double [] values1 = instance1.toDoubleArray();
    double [] values2 = instance2.toDoubleArray();
    // Go through all attributes
    for (int i = 0; i < values1.length; i++) {
      if (i != m_classIndex) {
	diff = values1[i] - values2[i];
	distance += diff * diff;
      }
    }
    distance = Math.sqrt(distance);
    return distance;
  };

  /**
   * Returns a similarity estimate between two instances. Similarity is obtained by
   * inverting the distance value using one of three methods:
   * CONVERSION_LAPLACIAN, CONVERSION_EXPONENTIAL, CONVERSION_UNIT.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarity(Instance instance1, Instance instance2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + distance(instance1, instance2));
    case CONVERSION_UNIT:
      return 2 * (1 - distance(instance1, instance2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-distance(instance1, instance2));
    default:
      throw new Exception ("Unknown distance to similarity conversion method");
    }
  }

    /**
   * Returns a similarity estimate between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarityNonWeighted(Instance instance1, Instance instance2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + distanceNonWeighted(instance1, instance2));
    case CONVERSION_UNIT:
      return 2 * (1 - distanceNonWeighted(instance1, instance2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-distanceNonWeighted(instance1, instance2));
    default:
      throw new Exception ("Unknown distance to similarity conversion method");
    }
  }


  /** Get the values of the partial derivates for the metric components
   * for a particular instance pair
   @param instance1 the first instance
   @param instance2 the first instance
   */
  public double[] getGradients(Instance instance1, Instance instance2) throws Exception {
    double[] gradients = new double[m_numAttributes];
    double distance = distance(instance1, instance2);

    // gradients are zero for 0-distance instances
    if (distance == 0) {
      return gradients;
    }

    // take care of SparseInstances by enumerating over the values of the first instance
    for (int i = 0; i < m_numAttributes; i++) {
      // get the values 
      double val1 = instance1.valueSparse(i);
      Attribute attr = instance1.attributeSparse(i);
      double val2 = instance2.value(attr);
      gradients[i] = 1.0 / (2*distance) * (val2 - val1) * (val2 - val1);
    }
    return gradients;
  }
  

  /** Train the metric
   */
  public void learnMetric (Instances data) throws Exception {
    System.err.println("WeightedMahalanobis can only be learned by the outside algorithm");
  }

  /** Set the weights */
  public void setWeights(Matrix weights) {

//      System.out.println("Setting weights: ");
//      for (int i = 0; i < weights.getArray().length; i++) {
//        for (int j = 0; j < weights.getArray()[i].length; j++) {
//    	System.out.print((float)weights.getArray()[i][j] + "\t");
//        }
//        System.out.println();
//      }
    
    // check that the matrix is positive semi-definite
    boolean isPSD = true;
    EigenvalueDecomposition ed = weights.eig();
    Matrix eigenVectorsMatrix = ed.getV();
    double[][] eigenVectors = eigenVectorsMatrix.getArray();
    double[] evalues = ed.getRealEigenvalues();
    double [][] evaluesM = new double[evalues.length][evalues.length];
    for (int i = 0; i < evalues.length; i++) {
      if (evalues[i] < 0) {
	evaluesM[i][i] = 0;
	isPSD = false; 
      } else {
	evaluesM[i][i] = evalues[i];
      }
    }

    if (!isPSD) { 
      // update the weights:  A = V * E * V'
      Matrix eigenValuesMatrix = new Matrix(evaluesM);
      weights = (eigenVectorsMatrix.times(eigenValuesMatrix)).times(eigenVectorsMatrix.transpose());
    
//        System.out.println("NON-PSD MATRIX!  After zeroing negative eigenvalues got: ");
//        for (int i = 0; i < weights.getArray().length; i++) {
//  	for (int j = 0; j < weights.getArray()[i].length; j++) {
//  	  System.out.print((float)weights.getArray()[i][j] + "\t");
//  	}
//  	System.out.println();
//        }
    }

    m_weightsMatrix = weights.getArray();

    // if the matrix is singular, need to be careful with m_weightsMatrixSquare and not use Cholesky
    if (!isPSD || weights.det() < Math.pow(10, -2* m_weightsMatrix.length)) {
      System.out.println("Singular weight matrix! det=" + weights.det());

      int maxIterations = 1000;
      int currIteration = 0;
      double det = weights.det();

      while (Math.abs(det) < 1.0e-8 && currIteration++ < maxIterations) {
	Matrix regularizer = Matrix.identity(m_weightsMatrix.length, m_weightsMatrix.length);
	regularizer = regularizer.times(weights.trace() * 0.01);
	weights = weights.plus(regularizer);  // W = W + 0.01tr(W) * I
	System.out.println("\t" + currIteration + ". det=" + ((float)det) + 
			   "\tafter FIXING AND REGULARIZATION det=" + weights.det());
	det = weights.det();
      }
      
      // if the matrix is irrepairable, use alternate factorization
      if (currIteration >= maxIterations) {
	System.out.println("IRREPAIRABLE MATRIX, using an alternate factorization:"); 
	// sqWeights = (Lambda+)^.5 * Q^T
	for (int i = 0; i < evaluesM.length; i++) {
	  evaluesM[i][i] = Math.sqrt(evaluesM[i][i]); 
	}
	m_weightsMatrixSquare = new double[m_weightsMatrix.length][m_weightsMatrix.length];
	for (int i = 0; i < m_weightsMatrixSquare.length; i++) {
	  for (int j = 0; j < m_weightsMatrixSquare[i].length; j++) {
	    m_weightsMatrixSquare[i][j] += eigenVectors[i][j] * evaluesM[j][j];
	  } 
	}
	//	m_weightsMatrixSquare = null;
	//  m_weightsMatrix = Matrix.identity(m_weightsMatrix.length, m_weightsMatrix.length).getArray();
//  	m_weightsMatrixSquare = Matrix.identity(m_weightsMatrix.length, m_weightsMatrix.length).getArray();
      } else { //  the matrix is positive definite, can do Cholesky
	m_weightsMatrix = weights.getArray();
	m_weightsMatrixSquare = weights.chol().getL().getArray();
      } 

//        System.out.println("\nsq weights: ");
//        for (int i = 0; i < m_weightsMatrixSquare.length; i++) {
//  	for (int j = 0; j < m_weightsMatrixSquare[i].length; j++) {
//  	  System.out.print(((float)m_weightsMatrixSquare[i][j]) + "\t");
//  	}
//  	System.out.println();
//        }
//        System.out.println("sqWeights*sqWeights'");
//        Matrix sqWeights = new Matrix(m_weightsMatrixSquare);
//        double[][] sanity = sqWeights.times(sqWeights.transpose()).getArray();
//        for (int i = 0; i < sanity.length; i++) {
//  	for (int j = 0; j < sanity[i].length; j++) {
//  	  System.out.print(((float)sanity[i][j]) + "\t");
//  	}
//  	System.out.println();
//        }
    
//        System.out.println("ACTUAL weights: ");
//        for (int i = 0; i < m_weightsMatrix.length; i++) {
//  	for (int j = 0; j < m_weightsMatrix[i].length; j++) {
//  	  System.out.print(((float)m_weightsMatrix[i][j]) + "\t");
//  	}
//  	System.out.println();
//        }
      
    } else {
      m_weightsMatrix = weights.getArray();
      m_weightsMatrixSquare = weights.chol().getL().getArray();
    }
    
    m_projectedInstanceHash = new HashMap();
    m_maxPoints = null;
    m_maxProjPoints = null;

    recomputeNormalizer();
    recomputeRegularizer();
  }

  /** override the parent class methods */
  public void setWeights(double[] weights) {
    int numAttributes = weights.length;
    m_weightsMatrix = new double[numAttributes][numAttributes];
    for (int i = 0; i < numAttributes; i++) {
      m_weightsMatrix[i][i] = weights[i];
    }
    setWeights(new Matrix(m_weightsMatrix));
  }

  /** override the parent class methods */
  public double[] getWeights() {
    double [] weights = new double[m_weightsMatrix.length];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = m_weightsMatrix[i][i];
    }
    return weights; 
  }

  /** override the parent class methods */
  public Matrix  getWeightsMatrix() {
    return new Matrix(m_weightsMatrix); 
  }

  /** Computes the regularizer */
  public void recomputeRegularizer() {
    Matrix weightMatrix = new Matrix(m_weightsMatrix);
    // TODONOW:  implement regularization
    m_regularizerVal = weightMatrix.norm1();
  }

  /** Computes the L1 regularizer */
  public void recomputeNormalizer() {
    Matrix weightMatrix = new Matrix(m_weightsMatrix);
    m_normalizer = Math.log(weightMatrix.det());
    System.out.println("Recomputed normalizer: " + (float) m_normalizer);
  }

  
  /**
   * Create an instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first instance
   * @param instance2 second instance
   */
  public Instance createDiffInstance (Instance instance1, Instance instance2) {
    double[] values1 = instance1.toDoubleArray();
    double[] values2 = instance2.toDoubleArray();
	
    int numAttributes = values1.length;
    if (instance1.classIndex() > 0) {
      numAttributes--;
    } 

    double[] diffInstanceValues = new double[numAttributes];  
    for (int i = 0; i < numAttributes; i++) {
      diffInstanceValues[i] =  (values1[i] - values2[i]);
    }
    Instance diffInstance = new Instance(1.0, diffInstanceValues);
    return diffInstance;
  }

  /**
   * Create a matrix of the form (inst1 - inst2) * (inst1 - inst2)^T
   * @param instance1 first instance
   * @param instance2 second instance
   */
  public Matrix createDiffMatrix (Instance instance1, Instance instance2) {
    Instance diffInstance = createDiffInstance(instance1, instance2);
    double [] values = diffInstance.toDoubleArray();
    int numAttributes = diffInstance.numValues();
    double [][] matrix = new double [numAttributes][numAttributes];
    for (int i = 0; i < numAttributes; i++) {
      for (int j = 0; j <= i; j++) {
	matrix[i][j] = values[i] * values[j];
	matrix[j][i] = matrix[i][j];
      } 
    }
    return new Matrix(matrix); 
  }

  /**
   * Set the type of  distance to similarity conversion. Values other
   * than CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL will be ignored
   * 
   * @param type type of the similarity to distance conversion to use
   */
  public void setConversionType(SelectedTag conversionType) {
    if (conversionType.getTags() == TAGS_CONVERSION) {
      m_conversionType = conversionType.getSelectedTag().getID();
    }
  }

  /**
   * return the type of distance to similarity conversion
   * @return one of CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL
   */
  public SelectedTag getConversionType() {
    return new SelectedTag(m_conversionType, TAGS_CONVERSION);
  }

    /** The computation of a metric can be either based on distance, or on similarity
   * @returns true because euclidean metrict fundamentally computes distance
   */
  public boolean isDistanceBased() {
    return true;
  }

  /**
   * Given a cluster of instances, return the centroid of that cluster
   * @param instances objects belonging to a cluster
   * @param fastMode whether fast mode should be used for SparseInstances
   * @param normalized normalize centroids for SPKMeans
   * @return a centroid instance for the given cluster
   */
  public Instance getCentroidInstance(Instances instances, boolean fastMode, boolean normalized) {
    System.err.println("Mahalanobis distance does not return a centroid");
    return null;
  }


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -N <br>
   * Normalize the euclidean distance by vectors lengths
   *
   * -E <br>
   * Use exponential conversion from distance to similarity
   * (default laplacian conversion) <p>
   *
   * -U <br>
   * Use unit conversion from similarity to distance (dist=1-sim)
   * (default laplacian conversion) <p>
   *
   * -R <br>
   * The metric is trainable and will be trained using the current MetricLearner
   * (default non-trainable)
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    if (Utils.getFlag('E', options)) {
      setConversionType(new SelectedTag(CONVERSION_EXPONENTIAL, TAGS_CONVERSION));
    } else if (Utils.getFlag('U', options)) {
      setConversionType(new SelectedTag(CONVERSION_UNIT, TAGS_CONVERSION));
    } else {
      setConversionType(new SelectedTag(CONVERSION_LAPLACIAN, TAGS_CONVERSION));
    }

    Utils.checkForRemainingOptions(options);
  }


  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(4);

    newVector.addElement(new Option("\tNormalize the euclidean distance by vectors lengths\n",
				    "N", 0, "-N"));
    newVector.addElement(new Option("\tUse exponential conversion from similarity to distance\n",
				    "E", 0, "-E"));
    newVector.addElement(new Option("\tUse unit conversion from similarity to distance\n",
				    "U", 0, "-U"));
    newVector.addElement(new Option("\tTrain the metric\n",
				    "R", 0, "-R"));
    newVector.addElement(new Option("\tUse the metric learner for similarity calculations(\"external\")",
				    "X", 0, "-X"));
    newVector.addElement(new Option(
	      "\tFull class name of metric learner to use, followed\n"
	      + "\tby scheme options. (required)\n"
	      + "\teg: \"weka.core.metrics.ClassifierMetricLearner -B weka.classifiers.function.SMO\"",
	      "L", 1, "-L <classifier specification>"));
    
    return newVector.elements();
  }

  /**
   * Gets the current settings of WeightedMahalanobisP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [45];
    int current = 0;

    if (m_conversionType == CONVERSION_EXPONENTIAL) {
      options[current++] = "-E";
    } else if (m_conversionType == CONVERSION_UNIT) {
      options[current++] = "-U";
    }
    
    if (m_trainable) {
      options[current++] = "-R";
    }
	
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /** Create a copy of this metric */
  public Object clone() {
    WeightedMahalanobis m = null; 
    m = (WeightedMahalanobis) super.clone();
    
    return m;
  }

    
  public static void main(String[] args) {
    try {
      // Create numeric attributes "length" and "weight"
      Attribute length = new Attribute("length");
      Attribute weight = new Attribute("weight");
      Attribute height = new Attribute("height");
      Attribute width = new Attribute("width");
      
      // Create vector to hold nominal values "first", "second", "third" 
      FastVector my_nominal_values = new FastVector(3); 
      my_nominal_values.addElement("first"); 
      my_nominal_values.addElement("second"); 
      my_nominal_values.addElement("third"); 
      
      // Create nominal attribute "position" 
      Attribute position = new Attribute("position", my_nominal_values);
      
      // Create vector of the above attributes 
      FastVector attributes = new FastVector(4);
      attributes.addElement(length);
      attributes.addElement(weight);
      attributes.addElement(height);
      attributes.addElement(width);
      attributes.addElement(position);
      
      // Create the empty dataset "race" with above attributes
      Instances race = new Instances("race", attributes, 0);
      
      // Make position the class attribute
      race.setClassIndex(position.index());
      
      // Create a sparse instance with three attribute values
      SparseInstance s_inst1 = new SparseInstance(1, new double[0], new int[0], 4);
      s_inst1.setValue(length, 2);
      s_inst1.setValue(weight, 1);
      s_inst1.setValue(position, "third");

      // Create a sparse instance with three attribute values
      SparseInstance s_inst2 = new SparseInstance(1, new double[0], new int[0], 4);
      s_inst2.setValue(length, 1);
      s_inst2.setValue(height, 5);
      s_inst2.setValue(position, "second");

      // Create a non-sparse instance with all attribute values
      Instance inst1 = new Instance(5);
      inst1.setValue(length, 3);
      inst1.setValue(weight, 4);
      inst1.setValue(height, 5);
      inst1.setValue(width, 2);
      inst1.setValue(position, "first");

      // Create a sparse instance with three attribute values
      Instance inst2 = new Instance(5);
      inst2.setValue(length, 2);
      inst2.setValue(weight, 2);
      inst2.setValue(height, 2);
      inst2.setValue(width, 3);
      inst2.setValue(position, "second");
      // Set instances' dataset to be the dataset "race"
      s_inst1.setDataset(race);
      s_inst2.setDataset(race);
      inst1.setDataset(race);
      inst2.setDataset(race);
      
      // Print the instances
      System.out.println("Sparse instance S1: " + s_inst1);
      System.out.println("Sparse instance S2: " + s_inst2);
      System.out.println("Non-sparse instance NS1: " + inst1);
      System.out.println("Non-sparse instance NS2: " + inst2);
      
      // Print the class attribute
      System.out.println("Class attribute: " + s_inst1.classAttribute());
      
      // Print the class index
      System.out.println("Class index: " + s_inst1.classIndex());

      // Create a new metric and print the distances
      WeightedMahalanobis metric = new WeightedMahalanobis(4);
      metric.setClassIndex(position.index());
      System.out.println("Distance between S1 and S2: " + metric.distance(s_inst1, s_inst2));
      System.out.println("Distance between S1 and NS1: " + metric.distance(s_inst1, inst1));
      System.out.println("Distance between NS1 and S1: " + metric.distance(inst1, s_inst1));
      System.out.println("Distance between NS1 and NS2: " + metric.distance(inst1, inst2));
      System.out.println("\nDistance-similarity conversion type: " +
			 metric.getConversionType().getSelectedTag().getReadable());
      System.out.println("Similarity between S1 and S2: " + metric.similarity(s_inst1, s_inst2));
      System.out.println("Similarity between S1 and NS1: " + metric.similarity(s_inst1, inst1));
      System.out.println("Similarity between NS1 and S1: " + metric.similarity(inst1, s_inst1));
      System.out.println("Similarity between NS1 and NS2: " + metric.similarity(inst1, inst2));
      System.out.println();
      System.out.println("Difference instance S1-S2: " + metric.createDiffInstance(s_inst1, s_inst2));
      System.out.println("Difference instance S1-NS1: " + metric.createDiffInstance(s_inst1, inst1));
      System.out.println("Difference instance NS1-S1: " + metric.createDiffInstance(inst1, s_inst1));
      System.out.println("Difference instance NS1-NS2: " + metric.createDiffInstance(inst1, inst2));
    } catch (Exception e) {
      e.printStackTrace();
    }	
  } 
}

