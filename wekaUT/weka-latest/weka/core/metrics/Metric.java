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
 *    Metric.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import java.util.ArrayList;
import java.io.Serializable;

import weka.core.*;

/** 
 * Abstract Metric class
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.4 $
 */

public abstract class Metric implements Cloneable, Serializable {
  /** number of attributes */
  protected int m_numAttributes = 0;

  /** indeces of attributes which the metric works on  */
  protected int [] m_attrIdxs =  null;

  /** index of the class attribute */
  protected int m_classIndex = -1;

  // ===============
  // Public methods.
  // ===============
 
  /**
   * Generates a new Metric with a specified number of
   * attributes. Has to initialize all fields of the metric with
   * default values.
   *
   * @param numAttributes the number of attributes that the metric will work on
   * @exception Exception if the distance metric has not been
   * generated successfully.  */
  public abstract void buildMetric(int numAttributes) throws Exception;

  /**
   * Generates a new Metric. Has to initialize all fields of the metric
   * with default values
   *
   * @param numAttributes the number of attributes that the metric will work on
   * @param options an array of options suitable for passing to setOptions.
   * May be null. 
   * @exception Exception if the distance metric has not been
   * generated successfully.
   */
  public abstract void buildMetric(int numAttributes, String[] options) throws Exception;

  /**
   * Create a new metric for operating on specified instances
   * @param data instances that the metric will be used on
   */
  public abstract void buildMetric(Instances data) throws Exception;
    
  /**
   * Specifies a list of attributes which will be used by the metric
   *
   * @param attrs an array of attribute indices
   */
  public void setAttrIdxs (int[] attrIdxs) {
    m_attrIdxs = new int[attrIdxs.length];
    System.arraycopy(attrIdxs, 0, m_attrIdxs, 0, attrIdxs.length);
  }

  /**
   * Returns an array of attribute incece which will be used by the metric
   *
   * @return an array of attribute indices
   */
  public int[] getAttrIndxs () {
    return m_attrIdxs;
  }

  /**
   * Specifies an interval of attributes which will be used by the metric
   *
   * @param begin_index beginning of attribute index interval
   * @param end_index end of attribute index interval
   */
  public void setAttrIdxs (int startIdx, int endIdx) {
    m_attrIdxs = new int[endIdx - startIdx + 1];
    for (int i = startIdx; i <= endIdx; i++)
      m_attrIdxs[i - startIdx] = i;
  }

   
  /**
   * Returns a distance value between two instances.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public abstract double distance(Instance instance1,
				  Instance instance2) throws Exception;


  /**
   * Returns a similarity estimate between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public abstract double similarity(Instance instance1,
				    Instance instance2) throws Exception;


  /** Return the penalty contribution -
   * typically equivalent to distance, distance^2, or -similarity */
  public abstract double penalty(Instance instance1,
				 Instance instance2) throws Exception;

  /** Return the penalty contribution that must be symmetric; 
   * typically equivalent to penalty, except for KL<>JS */
  public abstract double penaltySymmetric(Instance instance1,
					  Instance instance2) throws Exception;


  /**
   * Returns similarity value between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public abstract double similarityNonWeighted(Instance instance1,
					       Instance instance2) throws Exception;

  
  /**
   * Returns distance between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public abstract double distanceNonWeighted(Instance instance1,
					     Instance instance2) throws Exception;

  
  /**
   * It is often the case that last attribute of the data is the class.
   * This function takes instances, and returns an array of integers
   * 0..(num_attributes-1 - 1) to exclude the class attribute
   *
   * @return array of integer indeces of attributes, excluding
   * last one which is the class index
   */
  public int[] getAttrIdxsWithoutLastClass(Instances instances) {
    int [] attrIdxs;

    attrIdxs = new int[instances.numAttributes() - 1];
    for (int i = 0; i < attrIdxs.length; i++) {
      attrIdxs[i] = i;
    }
    return attrIdxs;
  }

  /**
   * This function takes instances, and returns an array of integers
   * 0..(num_attributes-1)
   *
   * @return array of integer indeces of attributes
   */
  public int[]  getAttrIdxs(Instances instances) {
    int [] attrIdxs;

    attrIdxs = new int[instances.numAttributes()];
    for (int i = 0; i < attrIdxs.length; i++) {
      attrIdxs[i] = i;
    }
    return attrIdxs ;
  }

  /** Specify which attribute is the class attribute
   * @param classAttrIdx the index of the class attribute
   */
  public void setClassIndex(int classIndex) {
    m_classIndex = classIndex;
  }

  /** Get the index of the attribute is the class attribute
   * @returns the index of the class attribute
   */
  public int getClassIndex(int classIndex) {
    return m_classIndex;
  }

  /** Get the number of attributes that the metric uses
   * @returns the number of attributes that the metric uses
   */
  public int getNumAttributes() {
    return m_numAttributes;
  }

  /** The computation of a metric can be either based on distance, or on similarity
   * @returns true if the underlying metric computes distance, false if similarity
   */
  public abstract boolean isDistanceBased();

  /** Get the norm-2 length of an instance assuming all attributes are numeric
   * @returns norm-2 length of an instance
   */
  public static double length(Instance instance) {
    int classIndex = instance.classIndex();
    double length = 0;
	
    if (instance instanceof SparseInstance) {
      // remap classIndex to an internal index
      if (classIndex >= 0) {
	classIndex = ((SparseInstance)instance).locateIndex(classIndex);
      }
      for (int i = 0; i < instance.numValues(); i++) {
	if (i != classIndex) {
	  double value = instance.valueSparse(i);
	  length += value * value;
	}
      }
    } else {  // non-sparse instance
      double[] values = instance.toDoubleArray(); 
      for (int i = 0; i < values.length; i++) {
	if (i != classIndex) {
	  length += values[i] * values[i];
	}
      }
    }
    return Math.sqrt(length);
  }


  /** Normalizes the values of a normal Instance
   *
   * @author Sugato Basu
   * @param inst Instance to be normalized
   */
  public void normalizeInstance(Instance inst) throws Exception{
    double norm = 0;
    double values [] = inst.toDoubleArray();

    if (inst instanceof SparseInstance) {
      throw new Exception("Use normalizeSparseInstance function");
    }

    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	norm += values[i] * values[i];
      }
    }
    norm = Math.sqrt(norm);
    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	values[i] /= norm;
      }
    }
    inst.setValueArray(values);
  }

  /**
   * Creates a new instance of a metric given it's class name and
   * (optional) arguments to pass to it's setOptions method. If the
   * classifier implements OptionHandler and the options parameter is
   * non-null, the classifier will have it's options set.
   *
   * @param metricName the fully qualified class name of the metric 
   * @param options an array of options suitable for passing to setOptions. May
   * be null.
   * @return the newly created metric ready for use.
   * @exception Exception if the metric  name is invalid, or the options
   * supplied are not acceptable to the metric 
   */
  public static Metric forName(String metricName,
			       String [] options) throws Exception {
    return (Metric)Utils.forName(Metric.class,
				 metricName,
				 options);
    }

  /** Create a copy of this metric */
  public Object clone() {
    Metric m = null; 
    try {
      m = (Metric) super.clone();
    } catch (CloneNotSupportedException e) {
      System.err.println("Metric can't clone");
    }
    // clone the fields
    if (m_attrIdxs != null) { 
      m.m_attrIdxs = (int []) m_attrIdxs.clone();
    }
    return m;
  } 
}





