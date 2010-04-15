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
 *    LearnableMetric.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import java.util.ArrayList;
import java.util.HashMap;

import weka.core.*;
import weka.classifiers.*;
import  weka.clusterers.regularizers.*;


/** 
 * Interface to distance metrics that can be learned
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.13 $
 */

public abstract class LearnableMetric extends Metric {
  /** Weights of individual attributes */
  protected double[] m_attrWeights = null;

  /** The metric may require normalizing all data */
  protected boolean m_normalizeData = false; 

  /** The maximum number of same-class examples to construct diff-instances from */
  protected int m_numPosDiffInstances = 200;

  /** Proportion of different-class versus same-class diff-instances */
  protected double m_posNegDiffInstanceRatio = 1.0;
  
  /** A metric may utilize a classifier for learning its parameters */
  boolean m_usesClassifier = false;
  protected String m_classifierClassName = null;
  public Classifier m_classifier = null;
  /** Certain classifiers may use non-nominal class attributes */
  protected boolean m_classifierRequiresNominalClass = false;

  /** True if metric learning is used.  Set to false to turn off metric learning */
  protected boolean m_trainable = false;
    
  /** Has the metric been trained? */
  boolean m_trained = false;

  /** True if metric uses an external estimator for calculating distances */
  boolean m_external = false;

  /** Current regularizer value */
  double m_regularizerVal = 0;

  /** use  regularization on weights */
  protected boolean m_regularize = false;
  protected Regularizer m_regularizer = new Rayleigh();


  /** Current normalizer value */
  double m_normalizer = 0;

  /** The metric may return its own maximum distance */
  public boolean m_fixedMaxDistance = false;
  protected double m_maxDistance = Double.MAX_VALUE;
  public double getMaxDistance() {
    return m_maxDistance;
  } 
  
  
  /**
   * Train the distance metric.  A specific metric will take care of
   * its own training either via a metric learner or by itself.
   */
  public abstract void learnMetric(Instances data) throws Exception;

  /**
   * switch from calculating the metric to pair-space classification
   *
   * @param classifierClassName Some classifier that classifies pairs of points
   * @param classifierRequiresNominalClass does classifier need a nominal class attribute?
   * Using DistributionClassifier because it actually reports a margin
   * SMO is first, will try others as well
   */
  public void useClassifier (String classifierClassName, boolean classifierRequiresNominalClass) throws Exception{
    m_classifierClassName = classifierClassName;
    m_classifier = (Classifier)Class.forName(classifierClassName).newInstance();
    m_classifierRequiresNominalClass = classifierRequiresNominalClass;
    m_usesClassifier = true;
  }

  /**
   * switch from using a classifier in difference-space to vanilla L-1
   * norm distance
   */
  public void useNoClassifier () {
    m_usesClassifier = false;
    m_classifier = null;
  }


  /**
   * Is this metric defined in vanilla space, or difference space?
   *
   * @return true if metric uses a classifier to classify L1-Norm as in-cluster
   * or out-of-cluster
   */
  public boolean usesClassifier() {
    return m_usesClassifier;
  }


  /** Is the metric normalizing? */
  public boolean doesNormalizeData() {
    return m_normalizeData; 
  } 
  
  /**
   * Reset all values that have been learned
   */
  public void resetMetric() throws Exception {
    m_trained = false;
    if (m_attrWeights != null) { 
      for (int i = 0; i < m_attrWeights.length; i++) {
	m_attrWeights[i] = 1;
      }
    }
    recomputeNormalizer();
    recomputeRegularizer();
  }

  /**
   * Create an instance with features corresponding to components of the two given instances
   * @param instance1 first instance
   * @param instance2 second instance
   */
  public abstract Instance createDiffInstance (Instance instance1, Instance instance2);


  /** Get the values of the partial derivates for the metric components
   * for a particular instance pair
   @param instance1 the first instance
   @param instance2 the first instance
   */
  public abstract double[] getGradients(Instance instance1, Instance instance2) throws Exception;
  
  /**
   * Set the feature weights
   *
   * @param weights an array of double weights for features
   */
  public void setWeights(double[] _weights) throws Exception{
    if (_weights.length != m_numAttributes) {
      throw new Exception("Number of weights " + _weights.length +
			  " is not equal to the number of attributes " + m_numAttributes);
    }
//      System.out.print("Setting weights: [  ");
//      for (int i = 0; i < _weights.length; i++) {
//        System.out.print(_weights[i] + "  ");
//        if (i > 100) {
//  	System.out.print("...");
//  	break;
//        }
//      }
//      System.out.println("]");
    
    m_attrWeights = new double[_weights.length];
    System.arraycopy(_weights, 0, m_attrWeights, 0, _weights.length);
    recomputeNormalizer();
    recomputeRegularizer();
  }

  /** Get the feature weights
   *
   * @return an array of feature weights
   */
  public double[] getWeights() {
    return m_attrWeights; 
  } 


  /**
   * Given a cluster of instances, return the centroid of that cluster
   * @param instances data points belonging to a cluster
   * @return a centroid instance for the given cluster
   */
  public abstract Instance getCentroidInstance(Instances instances, boolean fastMode, boolean normalized);


  /** Fast version of meanOrMode - streamlined from Instances.meanOrMode for efficiency 
   *  Does not check for missing attributes, assumes numeric attributes, assumes Sparse instances
   *  
   */
  public double[] meanOrMode(Instances insts) {

    int numAttributes = insts.numAttributes();
    double [] value = new double[numAttributes];
    double weight = 0;
    
    for (int i=0; i<numAttributes; i++) {
      value[i] = 0;
    }

    for (int j=0; j<insts.numInstances(); j++) {
      SparseInstance inst = (SparseInstance) (insts.instance(j));
      weight += inst.weight();
      for (int i=0; i<inst.numValues(); i++) {
	int indexOfIndex = inst.index(i);
	value[indexOfIndex]  += inst.weight() * inst.value(indexOfIndex);
      }
    }
    
    if (Utils.eq(weight, 0)) {
      for (int k=0; k<numAttributes; k++) {
	value[k] = 0;
      }
    }
    else {
      for (int k=0; k<numAttributes; k++) {
	value[k] = value[k] / weight;
      }
    }

    return value;
  }

  
  /**
   * Get the value of metricTraining
   *
   * @return Value of metricTraining
   */
  public boolean getTrainable() {
    
    return m_trainable;
  }
  
  /**
   * Set the value of metricTraining
   *
   * @param metricTraining Value of metricTraining
   */
  public void setTrainable(boolean metricTraining) {
    m_trainable = metricTraining;
  }

  /**
   * Get the value of m_external
   *
   * @return Value of m_external
   */
  public boolean getExternal() {
    
    return m_external;
  }
  
  /**
   * Set the value of m_external
   *
   * @param external if true, an external estimator will be used for distance
   */
  public void setExternal(boolean external) {
    m_external = external;
  }


  
  /** Set the number of positive instances  to be used for training
   * @param numPosInstances the number amounts of positive examples (diff-instances)
   */
  public void setNumPosDiffInstances (int numPosInstances) {
    m_numPosDiffInstances = numPosInstances;
  }

  /** Set the number of positive instances  to be used for training
   * @param numPosInstances the number amounts of positive examples (diff-instances)
   */
  public int getNumPosDiffInstances () {
    return m_numPosDiffInstances;
  }
  


  /** Set the ratio of positive and negative instances  to be used for training
   * @param ratio the relative amounts of negative examples compared to positive examples.
   * If -1, all possible negatives will be used (use with care!)
   */
  public void setPosNegDiffInstanceRatio (double ratio) {
    m_posNegDiffInstanceRatio = ratio;
  }

  /** Get the ratio of positive and negative instances  to be used for training
   * @returns the relative amounts of negative examples compared to positive examples.
   * If -1, all possible negatives will be used (use with care!)
   */
  public double getPosNegDiffInstanceRatio () {
    return m_posNegDiffInstanceRatio;
  }

  /** get the regularizer value */
  public double regularizer() {
    return m_regularizerVal;
  }

  /** recompute the normalizer - L1 by default; children may override */
  public void recomputeRegularizer() {
    m_regularizerVal = m_regularizer.computeRegularizer(m_attrWeights);
    
//      for (int i = 0; i < m_attrWeights.length; i++) {
//        if (m_attrWeights[i] != 0) { 
//  	m_regularizer += 1/Math.abs(m_attrWeights[i]); 
//  	// Removed, since this would encourage making some weights 0, which we want
//  	//        } else {
//  	//  	m_regularizer = Double.MAX_VALUE;
//        }
//      }
  }

  /** get the normalizer value */
  public double getNormalizer() {
    return m_normalizer;
  }

  /** recompute the normalizer - L1 by default; children may override */
  public void recomputeNormalizer() {
    m_normalizer = 0; 
    for (int i = 0; i < m_attrWeights.length; i++) {
      if (m_attrWeights[i] > 0) { 
	m_normalizer += Math.log(m_attrWeights[i]);
      } 
    }
  }

  
  /** Normalizes the values of an Instance utilizing feature weights
   *
   * @param inst Instance to be normalized
   */
  public void normalizeInstanceWeighted(Instance inst) {
    if (inst instanceof SparseInstance) {
      double norm = 0;
      int classIndex = inst.classIndex();
      for (int i = 0; i < inst.numValues(); i++) {
	int idx = inst.index(i);
	if (idx != classIndex) { // don't normalize the class index
	  norm += m_attrWeights[idx] * inst.value(idx) * inst.value(idx);
	}
      }

      norm = Math.sqrt(norm);
      //      System.out.println("norm: " + norm);
      for (int i = 0; i < inst.numValues(); i++) {
	int idx = inst.index(i);
	if (idx != classIndex) { 
	  inst.setValueSparse(i, inst.value(idx)/norm);
	}
      }
    } else { // non-sparse instances
      double norm = 0;
      double values [] = inst.toDoubleArray();
      int classIndex = inst.classIndex();

      for (int i=0; i<values.length; i++) {
	if (i != classIndex) { // don't normalize the class index 
	  norm += m_attrWeights[i] * values[i] * values[i];
	}
      }
      norm = Math.sqrt(norm);
      for (int i=0; i<values.length; i++) {
	if (i != classIndex) { // don't normalize the class index 
	  values[i] /= norm;
	}
      }
      inst.setValueArray(values);
    }
  }

  /** Set/get the regularizer */
  public void setRegularizer(Regularizer reg) {
    m_regularizer = reg;
  }
  public Regularizer getRegularizer() {
    return m_regularizer;
  }
  

  public static Metric forName(String metricName,
					String [] options) throws Exception {
    return (LearnableMetric)Utils.forName(LearnableMetric.class,
					  metricName,
					  options);
  }

  /** Create a copy of this metric */
  public Object clone() {
    LearnableMetric m = null; 
    m = (LearnableMetric) super.clone();
    
    // clone the fields
    if (m_attrWeights != null) { 
      m.m_attrWeights = (double []) m_attrWeights.clone();
    }
    if (m_classifier != null) {
      try { 
	m.m_classifier = Classifier.makeCopies(m_classifier, 1)[0];
      } catch (Exception e) {
	System.err.println("Problems cloning a non-null classifier in LearnableMetric; this should never be reached");
      }
    } 
    return m;
  }
}
