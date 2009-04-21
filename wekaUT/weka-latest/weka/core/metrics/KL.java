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
 *    KL.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import weka.core.*;
import weka.deduping.metrics.HashMapVector;

import java.util.*;

import java.io.*;


/** 
 * KL class
 *
 * Implements weighted Kullback-Leibler divergence 
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.23 $
 */

public class KL extends SmoothingMetric
  implements InstanceConverter, OptionHandler {
  public final double LOG2 = Math.log(2);

  /** We can switch between regular KL divergence and I-divergence */
  protected boolean m_useIDivergence = true;

  /** Frequencies over the entire dataset used for smoothing */
  protected HashMapVector m_datasetFrequencies = null;

  /** We hash sum(p log(p)) terms for the input instances to speed up computation */
  protected HashMap m_instanceNormHash = null; 

  /** Total number of tokens in the dataset */
  protected int m_numTotalTokens = 0;

  /** Different smoothing methods for obtaining probability distributions from frequencies  */
  public static final int SMOOTHING_UNSMOOTHED = 1;
  public static final int SMOOTHING_DIRICHLET = 2;
  public static final int SMOOTHING_JELINEK_MERCER = 4;
  public static final Tag[] TAGS_SMOOTHING = {
    new Tag(SMOOTHING_UNSMOOTHED, "unsmoothed"),
    new Tag(SMOOTHING_DIRICHLET, "Dirichlet"),
    new Tag(SMOOTHING_JELINEK_MERCER, "Jelinek-Mercer")
      };
  /** The smoothing method */
  protected int m_smoothingType = SMOOTHING_UNSMOOTHED;
 
  /** The pseudocount value for the Dirichlet smoothing */
  protected double m_pseudoCountDirichlet = 1.0;

  /** The lambda value for the Jelinek-Mercer smoothing */
  protected double m_lambdaJM = 0.5;
     
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

  /** A metric learner responsible for training the parameters of the metric */
  protected MetricLearner m_metricLearner = new ClassifierMetricLearner();
//    protected MetricLearner m_metricLearner = new GDMetricLearner();



  /** A hashmap that maps every instance to a set of instances with which JS has been computed */
  protected HashMap m_instanceConstraintMap = new HashMap();
  
  /**
   * Create a new metric.
   * @param numAttributes the number of attributes that the metric will work on
   */ 
  public KL(int numAttributes) throws Exception {
    super(); 
    buildMetric(numAttributes);
  }

  /** Create a default new metric */
  public KL() {
    m_fixedMaxDistance = true; 
    m_maxDistance = 1; 
  } 
   
  /**
   * Creates a new metric which takes specified attributes.
   *
   * @param _attrIdxs An array containing attribute indeces that will
   * be used in the metric
   */
  public KL(int[] _attrIdxs) throws Exception {
    super();
    setAttrIdxs(_attrIdxs);
    buildMetric(_attrIdxs.length);	
  }

  /**
   * Reset all values that have been learned
   */
  public void resetMetric() throws Exception {
    super.resetMetric();
    m_currAlpha = m_alpha;
    m_instanceConstraintMap = new HashMap();
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
    m_attrWeights = new double[numAttributes];
    m_attrIdxs = new int[numAttributes];
    for (int i = 0; i < numAttributes; i++) {
      m_attrWeights[i] = 1;
      m_attrIdxs[i] = i;
    }
    m_instanceConstraintMap = new HashMap();

    m_currAlpha = m_alpha;
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
    buildMetric(m_numAttributes);

    // hash the dataset-wide frequencies
    m_datasetFrequencies = new HashMapVector(); //  # of occurrences of each unique token in dataset
    m_numTotalTokens = 0;  // total # of (non-unique) tokens in the dataset
    double []instanceLengths = new double[data.numInstances()]; // num tokens per instance
    
    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i);
      for (int j = 0; j < instance.numValues(); j++) {
	Attribute attr = instance.attributeSparse(j);
	int attrIdx = instance.index(j);
	if (attrIdx != m_classIndex) {
	  m_datasetFrequencies.increment(attr.name(), instance.value(attr));
	  m_numTotalTokens += instance.value(attr);
	  instanceLengths[i] += instance.value(attr);
	}
      }
    }

    // convert all instances in the dataset
    System.out.println("\n\nConverting all instances for KL distance\n");
    Instances convertedData = new Instances(data, data.numInstances());
    for (int i = 0; i < data.numInstances(); i++) {
      convertedData.add(convertInstance(data.instance(i)));
      if (i % 10 == 9 ) System.out.print(".");
      if (i % 100 == 99) System.out.println(" " + (i+1));      
    }
    System.out.println();
    
    // copy all converted instances to original data
    data.delete();
    for (int i = 0; i < convertedData.numInstances(); i++) {
      data.add(convertedData.instance(i));
    }

    // Hash instance norms
    m_instanceNormHash = new HashMap();
    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i);
      double norm = 0;

      for (int j = 0; j < instance.numValues(); j++) { 
	int attrIdx = instance.index(j);
	if (attrIdx != m_classIndex) {
	  double value = instance.value(attrIdx);
	  norm += value * Math.log(value);
	}
      }
      m_instanceNormHash.put(instance, new Double(norm));
    }

    if (m_trainable) {
      learnMetric(data);
    }
  }

  public Instance convertInstance(Instance oldInstance) {
    Instance newInstance;
    if (oldInstance instanceof SparseInstance
	&& m_smoothingType == SMOOTHING_UNSMOOTHED) {
      newInstance = new SparseInstance(oldInstance);
    } else { // either original data is dense, or smoothing is on - returning dense.
      newInstance = new Instance(oldInstance.numAttributes());
    }
    newInstance.setDataset(oldInstance.dataset());
    int classIdx = oldInstance.classIndex();

    // get the total count of tokens for this instance
    double numTotalTokens = 0;
    for (int i = 0; i < oldInstance.numValues(); i++) {
      int idx = oldInstance.index(i);
      if (idx != classIdx) { 
	numTotalTokens += oldInstance.valueSparse(i);
      }
    }

    // we're iterating over newInstance in case
    // there was a transition from sparse to non-sparse. 
    for (int i = 0; i < newInstance.numValues(); i++) {
      int idx = newInstance.index(i);
      if (idx != classIdx) {
	Attribute attr = newInstance.attribute(idx);
	newInstance.setValue(idx, convertFrequency(oldInstance.value(idx),
						   numTotalTokens, attr.name()));
      }
    }
    return newInstance;
  } 

  /** Given a frequency of a given token in a document, convert 
   *  it to a probability value for that document's distribution
   * @param freq frequency of a token
   * @param token the token
   * @returns a probability value
   */
  protected double convertFrequency(double freq, double numTotalTokens, String token) {
    double datasetProb = 0; 
    switch (m_smoothingType) {
    case SMOOTHING_UNSMOOTHED:
      return freq/numTotalTokens;
    case SMOOTHING_DIRICHLET:
      datasetProb = m_datasetFrequencies.getWeight(token) / m_numTotalTokens;
      return (freq + m_pseudoCountDirichlet * datasetProb) /
	(numTotalTokens + m_pseudoCountDirichlet);
    case SMOOTHING_JELINEK_MERCER:
      datasetProb = m_datasetFrequencies.getWeight(token) / m_numTotalTokens;
      return (1 - m_lambdaJM) * (freq / numTotalTokens) + m_lambdaJM * datasetProb;
    default:
      System.err.println("Unknown smoothing method: " + m_smoothingType); 
      return -1; 
    }
  }

  /** Smooth an instance */
  public Instance smoothInstance(Instance instance) {
    int numAttributes = instance.numAttributes(); 
    double[] values = new double[numAttributes];
    double prior = 1.0/ numAttributes;
    for (int j = 0; j < numAttributes; j++) {
      values[j] = 1.0 / (1 + m_alpha) * (instance.value(j) +  m_alpha * prior);
    }

    return  new Instance(1.0, values); 
  } 
    
  
  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distance(Instance instance1, Instance instance2) throws Exception {
    // either pass the computation to the external classifier, or do the work yourself
    if (m_trainable && m_external && m_trained) {
      return m_metricLearner.getDistance(instance1, instance2);
    } else {
      return distanceInternal(instance1, instance2);
    }
  }

  /** Return the penalty contribution - KL */
  public double penalty(Instance instance1,
			Instance instance2) throws Exception {
    double distance = distance(instance1, instance2);
    return distance; 
  }

  /** Return the penalty contribution - JS */
  public double penaltySymmetric(Instance instance1,
			Instance instance2) throws Exception {
    double distance = distanceJS(instance1, instance2);
    return distance; 
  }


  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceInternal(Instance instance1, Instance instance2) throws Exception {
    if (instance1 instanceof SparseInstance) {
      return distanceSparse((SparseInstance)instance1, instance2);
    } else {
      return distanceNonSparse(instance1, instance2);
    }
  }
    

  /** Returns a distance value between two sparse instances. 
   * @param instance1 First sparse instance.
   * @param instance2 Second sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceSparse(SparseInstance instance1, Instance instance2) throws Exception {
    double distance = 0, value1, value2, idivTerm = 0;
    int numValues1 = instance1.numValues();
    boolean dbg = true;
	    
    // iterate through the attributes that are present in the first instance
    for (int i = 0; i < numValues1; i++) {
      int attrIdx = instance1.index(i);
      if (attrIdx != m_classIndex) {
	value1 = instance1.valueSparse(i);
	value2 = instance2.value(attrIdx);
	if (value2 > 0) { 
	  distance += m_attrWeights[attrIdx] * value1 * Math.log(value1/value2);
	  if (m_useIDivergence) {
	    idivTerm -= m_attrWeights[attrIdx] * value1;
	  }
	  //	    if (dbg) { System.out.println("\t" + attrIdx + "\t" + value1 + "  " + value2 + "\t" + distance); dbg= false;}
	} else {
	  System.err.println("KL.distanceNonSparse:  0 value in instance2, attribute=" + attrIdx + "\n" + instance2.value(attrIdx) + "\n" + instance2); 	  
	  return Double.MAX_VALUE;
	} 
      }
    }

    // if i-divergence is used, need to pick up values of instance2
    if (m_useIDivergence) { 
      for (int i = 0; i < instance2.numValues(); i++) {
	int attrIdx = instance2.index(i);
	if (attrIdx != m_classIndex) {
	  value2 = instance2.valueSparse(i);
	  idivTerm += m_attrWeights[attrIdx] * value2;
	}
      }
    }

    distance = distance + idivTerm;
    return distance;
  }


  /** Returns a distance value between non-sparse instances without using the weights
   * @param instance1 non-sparse instance.
   * @param instance2 non-sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceNonSparse(Instance instance1, Instance instance2) throws Exception {
    double distance = 0, idivTerm = 0;
    double [] values1 = instance1.toDoubleArray();
    double [] values2 = instance2.toDoubleArray();
    // Go through all attributes
    for (int i = 0; i < values1.length; i++) {
      if (i != m_classIndex) {
	if (values2[i] > 0) { 
	  distance += m_attrWeights[i] * (values1[i] * Math.log(values1[i]/values2[i]));
	  if (m_useIDivergence) {
	    idivTerm -= m_attrWeights[i] * (values1[i] - values2[i]);
	  }
	} else {  // instance2 has a 0 value
	  System.err.println("KL.distanceNonSparse:  0 value in instance2, attribute=" + i); 
	  return  Double.MAX_VALUE;
	}
      }
    }
    distance = distance + idivTerm;
    return distance;
  };


  /**
   * Returns Jensen-Shannon distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distanceJS could not be estimated.
   */
  public double distanceJS(Instance instance1, Instance instance2) throws Exception {
    if (instance1 instanceof SparseInstance && instance2 instanceof SparseInstance) {
      return distanceJSSparse((SparseInstance)instance1, (SparseInstance)instance2);
    } else {
      return distanceJSNonSparse(instance1, instance2);
    }
  }
    

  /** Returns Jensen-Shannon distance between two sparse instances. 
   * @param instance1 First sparse instance.
   * @param instance2 Second sparse instance.
   * @exception Exception if distanceJS could not be estimated.
   */
  public double distanceJSSparse(SparseInstance instance1,
				 SparseInstance instance2) throws Exception {
    double distanceJS = 0;
    boolean lookupOK = false; 
    if (m_instanceConstraintMap.containsKey(instance1)) {
      HashMap instanceDiffInstanceMap =
	(HashMap) m_instanceConstraintMap.get(instance1);
      if (instanceDiffInstanceMap.containsKey(instance2)) {
	lookupOK = true;
	SparseInstance diffVector = (SparseInstance) instanceDiffInstanceMap.get(instance2);
	for (int i = 0; i < diffVector.numValues(); i++) {
	  int idx = diffVector.index(i);
	  distanceJS += m_attrWeights[idx] * diffVector.valueSparse(i);
	} 
      } 
    }
    if (!lookupOK) { 
      double value1, value2, sum1 = 0, sum2 = 0;
      int numValues1 = instance1.numValues();
      int numValues2 = instance2.numValues();
      int maxNumValues = numValues1 + numValues2;  // the overall number of attributes
      double [] attrValues = new double[maxNumValues];
      int [] indices = new int[maxNumValues];
      Arrays.fill(attrValues, 0);
      Arrays.fill(indices, Integer.MAX_VALUE);

      // pick up the values from instance2 that didn't occur in instance1
      int counter = 0, counter1 = 0, counter2 = 0;
      int attrIdx1 = 0, attrIdx2 = 0;
      while (counter1 < numValues1 || counter2 < numValues2) {
	if (counter1 < numValues1) {
	  attrIdx1 = instance1.index(counter1);
	} else {
	  attrIdx1 = Integer.MAX_VALUE;
	}

	if (counter2 < numValues2) {
	  attrIdx2 = instance2.index(counter2);
	} else {
	  attrIdx2 = Integer.MAX_VALUE;
	}

	while (attrIdx1 < attrIdx2 && counter1 < numValues1 ) {
	  if (attrIdx1 != m_classIndex) { 
	    sum1 += m_attrWeights[attrIdx1] * instance1.valueSparse(counter1);
	    attrValues[counter] = 0.5 * instance1.valueSparse(counter1);
	    indices[counter] = attrIdx1;
	    counter++;
	  }
	  counter1++;
	  if (counter1 < numValues1) { 
	    attrIdx1 = instance1.index(counter1);
	  }
	}

	while (attrIdx2 < attrIdx1 && counter2 < numValues2 ) {
	  if (attrIdx2 != m_classIndex) { 
	    sum2 += m_attrWeights[attrIdx2] * instance2.valueSparse(counter2);
	    attrValues[counter] = 0.5 * instance2.valueSparse(counter2);
	    indices[counter] = attrIdx2;
	    counter++;
	  }
	  counter2++;
	  if (counter2 < numValues2) {
	    attrIdx2 = instance2.index(counter2);
	  }
	}

	if (attrIdx1 == attrIdx2 && attrIdx1 != m_classIndex && attrIdx1 < Integer.MAX_VALUE && attrIdx2 < Integer.MAX_VALUE) {
	  value1 = instance1.valueSparse(counter1);
	  value2 = instance2.valueSparse(counter2);
	  distanceJS += m_attrWeights[attrIdx1] * (value1 * Math.log(value1) + value2 * Math.log(value2) 
						   - (value1 + value2) * Math.log((value1+value2)/2.0));
	  attrValues[counter] = 0.5 * (value1 * Math.log(value1) + value2 * Math.log(value2) -
				       (value1 + value2) * Math.log((value1+value2)/2.0)) / LOG2; 
	  indices[counter] = attrIdx1;
	  counter++;
	  counter1++;
	  counter2++;
	} else if (attrIdx1 == m_classIndex) {
	  if (instance1.classValue() == instance2.classValue()) {
	    attrValues[counter] = 1;
	  } else {
	    attrValues[counter] = -1;
	  }
	  indices[counter] = m_classIndex;
	  counter++;
	  counter1++;
	  counter2++;
	} 
      }

      SparseInstance diffInstanceJS = new SparseInstance(1.0, attrValues, indices, instance1.dataset().numAttributes());
      diffInstanceJS.setDataset(instance1.dataset());

      // hash the diff-instance for both instances involved
      HashMap instanceDiffInstanceMap1;
      if (m_instanceConstraintMap.containsKey(instance1)) {
	instanceDiffInstanceMap1 = (HashMap) m_instanceConstraintMap.get(instance1);
      } else { 
	instanceDiffInstanceMap1 = new HashMap();
	m_instanceConstraintMap.put(instance1, instanceDiffInstanceMap1);
      } 
      instanceDiffInstanceMap1.put(instance2, diffInstanceJS);

      HashMap instanceDiffInstanceMap2;
      if (m_instanceConstraintMap.containsKey(instance2)) {
	instanceDiffInstanceMap2 = (HashMap) m_instanceConstraintMap.get(instance2);
      } else { 
	instanceDiffInstanceMap2 = new HashMap();
	m_instanceConstraintMap.put(instance2, instanceDiffInstanceMap2);
      } 
      instanceDiffInstanceMap2.put(instance1, diffInstanceJS);

      distanceJS = 0.5 * (sum1 + sum2 + distanceJS / LOG2);
      if (distanceJS > 1.00001)  System.out.println("TROUBLE: distanceJS=" + distanceJS + " sum1=" + sum1 + " sum2=" + sum2);
    }
    return distanceJS;
  }

  /** Returns Jensen-Shannon distance between a non-sparse instance and a sparse instance
   * @param instance1 sparse instance.
   * @param instance2 sparse instance.
   * @exception Exception if distanceJS could not be estimated.
   */
  public double distanceJSSparseNonSparse(SparseInstance instance1, Instance instance2) throws Exception {
    double diff, distanceJS = 0, sum2 = 0;
    int numValues1 = instance1.numValues();
    int numValues2 = instance2.numValues();
    double [] values2 = instance2.toDoubleArray();

    // add all contributions of the second instance; unnecessary ones will be subtracted later
    for (int i = 0; i < values2.length; i++) {
      if (i != m_classIndex) {
	sum2 += m_attrWeights[i] * values2[i];
      }
    }

    for (int i = 0; i < numValues1; i++) {
      int attrIdx = instance1.index(i);

      if (attrIdx != m_classIndex) {
	double value1 = instance1.valueSparse(i);
	double value2 = values2[attrIdx];
	if (value1 != 0 && value2 != 0) { 
	  distanceJS += m_attrWeights[attrIdx] * (value1 * Math.log(value1) + value2 * Math.log(value2) 
						  - (value1 + value2) * Math.log((value1+value2)/2.0));
	  sum2 -= m_attrWeights[attrIdx] * value2;  // subtract the contribution previously added
	} 
      }
    } 
    distanceJS = 0.5 * (sum2 + distanceJS / LOG2);
    return distanceJS;
  }


  /** Returns Jensen-Shannon distance between non-sparse instances without using the weights
   * @param instance1 non-sparse instance.
   * @param instance2 non-sparse instance.
   * @exception Exception if distanceJS could not be estimated.
   */
  public double distanceJSNonSparse(Instance instance1, Instance instance2) throws Exception {
    double distanceJS = 0, sum1 = 0, sum2 = 0;
    double [] values1 = instance1.toDoubleArray();
    double [] values2 = instance2.toDoubleArray();
    // Go through all attributes
    for (int i = 0; i < values1.length; i++) {
      if (i != m_classIndex) {
	if (values1[i] != 0 && values2[i] != 0) {
	  distanceJS += m_attrWeights[i] * (values1[i] * Math.log(values1[i]) + values2[i] * Math.log(values2[i]) 
					    - (values1[i] + values2[i]) * Math.log((values1[i]+values2[i])/2.0));
	} else if (values1[i] != 0) {
	  sum1 += m_attrWeights[i] * values1[i];
	} else if (values2[i] != 0) {
	  sum2 += m_attrWeights[i] * values2[i];
	}
      }
    }
    distanceJS = 0.5 * (sum1 + sum2 + distanceJS / LOG2);
    return distanceJS;
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
   * Returns distance between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double distanceNonWeighted(Instance instance1, Instance instance2) throws Exception {
    return distance(instance1, instance2);
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
    double distance = distanceInternal(instance1, instance2);

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
      // TODO: why was this gradient used earlier??
      //      gradients[i] = 1.0 / (2*distance) * (val2 - val1) * (val2 - val1);
      if (val2 > 0) {
	gradients[i] = val1 * Math.log(val1/val2) - (val1 - val2);
      }
    }
    return gradients;
  }

  /** get the normalizer value */
  public double getNormalizer() {
    return 0;
  }
  

  /** Train the metric
   */
  public void learnMetric (Instances data) throws Exception {
    if (m_metricLearner == null) {
      System.err.println("Metric learner for KL is not initalized. No training was conducted");
      return;
    }
    // need Idivergence to be switched on during metric learning
    if (m_useIDivergence == false) {
      System.out.println("Using IDvergence ...");
      m_useIDivergence = true;
    }
    m_metricLearner.trainMetric(this, data);
  }

  /**
   * Set the distance metric learner
   *
   * @param metricLearner the metric learner
   */
  public void setMetricLearner (MetricLearner metricLearner) {
    m_metricLearner = metricLearner;
  }
 
  /**
   * Get the distance metric learner
   *
   * @returns the distance metric learner that this metric employs
   */
  public MetricLearner getMetricLearner () {
    return m_metricLearner;
  }
 
    
  /**
   * Create an instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first instance
   * @param instance2 second instance
   */
  public Instance createDiffInstance (Instance instance1, Instance instance2) {
    if (instance1 instanceof SparseInstance && instance2 instanceof SparseInstance) {
      return  createDiffInstanceSparse((SparseInstance)instance1, (SparseInstance)instance2);
    } else if (instance1 instanceof SparseInstance) {
      return createDiffInstanceSparseNonSparse((SparseInstance)instance1, instance2);
    }  else if (instance2 instanceof SparseInstance) {
      return createDiffInstanceSparseNonSparse((SparseInstance)instance2, instance1);
    } else {
      return createDiffInstanceNonSparse(instance1, instance2);
    }
  }

  /**
   * Create a sparse instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first sparse instance
   * @param instance2 second sparse instance
   */
  protected SparseInstance createDiffInstanceSparse (SparseInstance instance1, SparseInstance instance2) {
    int numValues1 = instance1.numValues();
    int numValues2 = instance2.numValues();
    int maxNumValues =numValues1 + numValues2;  // the overall number of attributes
    int classIndex = instance1.classIndex();
    
    int counter = 0;
    double [] attrValues = new double[maxNumValues];
    int [] indices = new int[maxNumValues];
    Arrays.fill(attrValues, 0);
    Arrays.fill(indices, Integer.MAX_VALUE);

    for (int i = 0; i < numValues1; i++) {
      int attrIdx = instance1.index(i);
      indices[counter] = attrIdx;

      if (attrIdx != classIndex) {  // skip class attributes
	double value1 = instance1.valueSparse(i);

	int idx2 = instance2.locateIndex(attrIdx);
	if (idx2 >=0 && attrIdx == instance2.index(idx2)) {
	  double value2 = instance2.valueSparse(idx2);
	  if (value2 > 0) { 
	    attrValues[counter] = value1 * Math.log(value1/value2) / LOG2;
	    if (m_useIDivergence) {
	      attrValues[counter] +=  value2; 
	    }
	  }
	}
	
	if (m_useIDivergence) {
	  attrValues[counter] -=  value1;
	}

      } else { // add the class value
	if (instance1.classValue() == instance2.classValue()) {
	  attrValues[counter] = 1;
	} else {
	  attrValues[counter] = -1;
	}
      }
      counter++;
    }

    SparseInstance diffInstance = new SparseInstance(1.0, attrValues, indices, instance1.dataset().numAttributes());
    diffInstance.setDataset(instance1.dataset());

    // if i-divergence is used, need to pick up "orphan" values of instance2
    if (m_useIDivergence) { 
      for (int i = 0; i < numValues2; i++) {
	int attrIdx = instance2.index(i);
	if (attrIdx != m_classIndex) {
	  double value2 = instance2.valueSparse(i);
	  int idx1 = instance1.locateIndex(attrIdx);
	  if (idx1 < 0 || attrIdx != instance1.index(idx1)) {
	    diffInstance.setValue(attrIdx, value2); 
	  }
	}
      }
    }

    return diffInstance;
  }

  /**
   * Create an instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first sparse instance
   * @param instance2 second non-sparse instance
   */
  protected Instance createDiffInstanceSparseNonSparse (SparseInstance instance1, Instance instance2) {
    int numValues1 = instance1.numValues();
    int numValues2 = instance2.numValues();
    Instance diffInstance = null;
    double[] values2 = instance2.toDoubleArray();
    int classIndex = instance1.classIndex();
    
    if (m_useIDivergence) {
      diffInstance = new Instance(instance2);   // "orphan values" from instance2 will be copied here
      for (int i = 0; i < numValues1; i++) {
	int idx = instance1.index(i); 
	if (idx != classIndex) {  // skip class attributes
	  double value1 = instance1.valueSparse(i);
	  if (values2[idx] != 0) { 
	    diffInstance.setValue(idx, value1 * Math.log(value1/values2[idx]) / LOG2 - value1);
	  } else {
	    System.err.println("\n\n\nPROBLEM!  Zero value in non-sparse instance in createDiffInstanceSparseNonSparse\n\n\n");
	    diffInstance.setValue(idx, -value1); 
	  } 
	} else { // add the class value
	  if (instance1.classValue() == instance2.classValue()) {
	    diffInstance.setClassValue(1);
	  } else {
	    diffInstance.setClassValue(-1);
	  }
	}
      }
    } else { // I-Divergence term is not used
      int counter = 0;
      double [] attrValues = new double[numValues1];
      int [] indices = new int[numValues1];
      Arrays.fill(attrValues, 0);
      Arrays.fill(indices, Integer.MAX_VALUE);

      for (int i = 0; i < numValues1; i++) {
	int attrIdx = instance1.index(i);
	indices[counter] = attrIdx;

	if (attrIdx != classIndex) {  // skip class attributes
	  double value1 = instance1.valueSparse(i);
	  if (values2[attrIdx] > 0) { 
	    attrValues[counter] = value1 * Math.log(value1/values2[attrIdx]);
	  }
	} else { // add the class value
	  if (instance1.classValue() == instance2.classValue()) {
	    attrValues[counter] = 1;
	  } else {
	    attrValues[counter] = -1;
	  }
	}
	counter++;
      }
      diffInstance = new SparseInstance(1.0, attrValues, indices, instance1.dataset().numAttributes());
    }
    diffInstance.setDataset(instance1.dataset());
    return diffInstance;
  }
    
  /**
   * Create a nonsparse instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first nonsparse instance
   * @param instance2 second nonsparse instance
   */
  protected Instance createDiffInstanceNonSparse (Instance instance1, Instance instance2) {
    double[] values1 = instance1.toDoubleArray();
    double[] values2 = instance2.toDoubleArray();
	
    int numAttributes = values1.length;
    // create an extra attribute if there was no class index originally
    int classIndex = instance1.classIndex();
    if (classIndex < 0) {
      classIndex = numAttributes;
      numAttributes++;
    } 
    double[] diffInstanceValues = new double[numAttributes];  

    for (int i = 0; i < values1.length; i++) {
      if (i != classIndex) {  // round up to float significance to be able to weed out duplicates later
	if (values1[i] != 0 && values2[i] != 0) { 
	  diffInstanceValues[i] = values1[i] * Math.log(values1[i]/values2[i]) / LOG2;
	  if (m_useIDivergence) {
	    diffInstanceValues[i] += values2[i] - values1[i];
	  }
	} else if (values2[i] != 0)  {
	  diffInstanceValues[i] = values2[i];
	} else if (values1[i] != 0) {
	  diffInstanceValues[i] = Double.MAX_VALUE;
	}
	
      } else {  // class value
	if (values1[i] == values2[i]) {
	  diffInstanceValues[i] = 1;
	} else {
	  diffInstanceValues[i] = -1;
	}
      }
    }
    Instance diffInstance = new Instance(1.0, diffInstanceValues);
    diffInstance.setDataset(instance1.dataset());
    return diffInstance;
  }

    /**
   * Create an instance with features corresponding to JS components
   * @param instance1 first instance
   * @param instance2 second instance
   */
  public Instance createDiffInstanceJS (Instance instance1, Instance instance2) {
    if (instance1 instanceof SparseInstance && instance2 instanceof SparseInstance) {
      return  createDiffInstanceJSSparse((SparseInstance)instance1, (SparseInstance)instance2);
    } else if (instance1 instanceof SparseInstance) {
      return createDiffInstanceJSSparseNonSparse((SparseInstance)instance1, instance2);
    }  else if (instance2 instanceof SparseInstance) {
      return createDiffInstanceJSSparseNonSparse((SparseInstance)instance2, instance1);
    } else {
      return createDiffInstanceJSNonSparse(instance1, instance2);
    }
  }

  /**
   * Create a sparse instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first sparse instance
   * @param instance2 second sparse instance
   */
  protected SparseInstance createDiffInstanceJSSparse (SparseInstance instance1, SparseInstance instance2) {
    SparseInstance diffInstanceJS = null;

    // try to look up this constraint in the hash
    if (m_instanceConstraintMap.containsKey(instance1)) {
      HashMap instanceDiffInstanceMap = (HashMap) m_instanceConstraintMap.get(instance1);
      if (instanceDiffInstanceMap.containsKey(instance2)) {
	diffInstanceJS = (SparseInstance) instanceDiffInstanceMap.get(instance2);
      }
    }

    // if the lookup failed, compute it and hash it
    if (diffInstanceJS == null)  { 
      System.err.println("\n\n\nThis should not happen!  Could not look up instance1 in createDiffInstanceJSSparse!\n\n\n\n");
      int numValues1 = instance1.numValues();
      int numValues2 = instance2.numValues();
      int maxNumValues = numValues1 + numValues2;  // the overall number of attributes
      double [] attrValues = new double[maxNumValues];
      int [] indices = new int[maxNumValues];
      Arrays.fill(attrValues, 0);
      Arrays.fill(indices, Integer.MAX_VALUE);

      int classIndex = instance1.classIndex();
      int counter = 0, counter1 = 0, counter2 = 0, attrIdx1, attrIdx2;

      while (counter1 < numValues1 || counter2 < numValues2) {
	if (counter1 < numValues1) {
	  attrIdx1 = instance1.index(counter1);
	} else {
	  attrIdx1 = Integer.MAX_VALUE;
	}

	if (counter2 < numValues2) {
	  attrIdx2 = instance2.index(counter2);
	} else {
	  attrIdx2 = Integer.MAX_VALUE;
	}

	while (attrIdx1 < attrIdx2 && counter1 < numValues1) {
	  if (attrIdx1 != m_classIndex) {
	    attrValues[counter] = 0.5 * instance1.valueSparse(counter1);
	    indices[counter] = attrIdx1;
	    counter++;
	  }
	  counter1++;
	  if (counter1 < numValues1) {
	    attrIdx1 = instance1.index(counter1);
	  }
	}
      
	while (attrIdx2 < attrIdx1 && counter2 < numValues2) {
	  if (attrIdx2 != m_classIndex) { 
	    attrValues[counter] = 0.5 * instance2.valueSparse(counter2);
	    indices[counter] = attrIdx2;
	    counter++;
	  }
	  counter2++;
	  if (counter2 < numValues2) { 
	    attrIdx2 = instance2.index(counter2);
	  }
	}
    
	if (attrIdx1 == attrIdx2 && attrIdx1 != m_classIndex && attrIdx1 < Integer.MAX_VALUE && attrIdx2 < Integer.MAX_VALUE) { 
	  double value1 = instance1.valueSparse(counter1);
	  double value2 = instance2.valueSparse(counter2);
	  attrValues[counter] = 0.5 * (value1 * Math.log(value1) + value2 * Math.log(value2) -
				       (value1 + value2) * Math.log((value1+value2)/2.0)) / LOG2; 
	  indices[counter] = attrIdx1;
	  counter++;
	  counter1++;
	  counter2++;
	} else if (attrIdx1 == m_classIndex) {
	  if (instance1.classValue() == instance2.classValue()) {
	    attrValues[counter] = 1;
	  } else {
	    attrValues[counter] = -1;
	  }
	  indices[counter] = m_classIndex;
	  counter++;
	  counter1++;
	  counter2++;
	} 
      }

      diffInstanceJS = new SparseInstance(1.0, attrValues, indices, instance1.dataset().numAttributes());
      diffInstanceJS.setDataset(instance1.dataset());

      // hash the diff-instance for both instances involved
      HashMap instanceDiffInstanceMap1;
      if (m_instanceConstraintMap.containsKey(instance1)) {
	instanceDiffInstanceMap1 = (HashMap) m_instanceConstraintMap.get(instance1);
      } else { 
	instanceDiffInstanceMap1 = new HashMap();
	m_instanceConstraintMap.put(instance1, instanceDiffInstanceMap1);
      } 
      instanceDiffInstanceMap1.put(instance2, diffInstanceJS);

      HashMap instanceDiffInstanceMap2;
      if (m_instanceConstraintMap.containsKey(instance2)) {
	instanceDiffInstanceMap2 = (HashMap) m_instanceConstraintMap.get(instance2);
      } else { 
	instanceDiffInstanceMap2 = new HashMap();
	m_instanceConstraintMap.put(instance2, instanceDiffInstanceMap2);
      } 
      instanceDiffInstanceMap2.put(instance1, diffInstanceJS);
    }
    return diffInstanceJS;
  }

  /**
   * Create an instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first sparse instance
   * @param instance2 second non-sparse instance
   */
  protected Instance createDiffInstanceJSSparseNonSparse (SparseInstance instance1, Instance instance2) {
    int numValues = instance2.numValues(); 
    int classIndex = instance1.classIndex();
    double [] attrValues = new double[numValues];
    double[] values2 = instance2.toDoubleArray();

    // add all contributions of the second instance; unnecessary ones will overwritten later
    for (int i = 0; i < values2.length; i++) {
      if (i != m_classIndex) {
	attrValues[i] = 0.5 * values2[i];
      }
    }

    for (int i = 0; i < instance1.numValues(); i++) {
      Attribute attribute = instance1.attributeSparse(i);
      int attrIdx = attribute.index();

      if (attrIdx != m_classIndex) {
	double value1 = instance1.value(attrIdx);
	double value2 = values2[attrIdx];
	if (value1 != 0 && value2 != 0) {
	  attrValues[attrIdx] = 0.5 * (value1 * Math.log(value1) + value2 * Math.log(value2) 
				       - (value1 + value2) * Math.log((value1+value2)/2.0)) / LOG2;
	} else if (value1 != 0) {
	  attrValues[attrIdx] = 0.5 * value1;
	} else if (value2 != 0) {
	  attrValues[attrIdx] = 0.5 * value2;
	} 
      } else { // class index
	if (instance1.classValue() == instance2.classValue()) {
	  attrValues[attrIdx] = 1;
	} else {
	  attrValues[attrIdx] = -1;
	}
      } 
    } 

    Instance diffInstanceJS = new Instance(1.0, attrValues);
    diffInstanceJS.setDataset(instance1.dataset());
    return diffInstanceJS;
  }
    
  /**
   * Create a nonsparse instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first nonsparse instance
   * @param instance2 second nonsparse instance
   */
  protected Instance createDiffInstanceJSNonSparse (Instance instance1, Instance instance2) {
    double[] values1 = instance1.toDoubleArray();
    double[] values2 = instance2.toDoubleArray();
	
    int numAttributes = values1.length;
    // create an extra attribute if there was no class index originally
    int classIndex = instance1.classIndex();
    if (classIndex < 0) {
      classIndex = numAttributes;
      numAttributes++;
    } 
    double[] diffInstanceJSValues = new double[numAttributes];  

    for (int i = 0; i < values1.length; i++) {
      if (i != classIndex) {  // round up to float significance to be able to weed out duplicates later
	if (values2[i] != 0 && values1[i] != 0) { 
	  diffInstanceJSValues[i] =  0.5 * (values1[i] * Math.log(values1[i]) + values2[i] * Math.log(values2[i])
					    - (values1[i] + values2[i]) * Math.log((values1[i]+values2[i])/2.0)) / LOG2;
	} else if (values1[i] != 0) {
	  diffInstanceJSValues[i] = 0.5 * values1[i];
	} else if (values2[i] != 0) {
	  diffInstanceJSValues[i] = 0.5 * values2[i];
	} 
      } else {  // class value
	if (values1[i] == values2[i]) {
	  diffInstanceJSValues[i] = 1;
	} else {
	  diffInstanceJSValues[i] = -1;
	}
      }
    }
    Instance diffInstanceJS = new Instance(1.0, diffInstanceJSValues);
    diffInstanceJS.setDataset(instance1.dataset());
    return diffInstanceJS;
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

  /**
   * Set the type of  smoothing
   * 
   * @param type type of smoothing
   */
  public void setSmoothingType(SelectedTag smoothingType) {
    if (smoothingType.getTags() == TAGS_SMOOTHING) {
      m_smoothingType = smoothingType.getSelectedTag().getID();
    }
  }

  /**
   * return the type of smoothing
   * @return one of SMOOTHING_UNSMOOTHED, SMOOTHING_DIRICHLET, SMOOTHING_JELINEK_MERCER
   */
  public SelectedTag getSmoothingType() {
    return new SelectedTag(m_smoothingType, TAGS_SMOOTHING);
  }

  /** Set the pseudo-count value for Dirichlet smoothing
   * @param pseudoCountDirichlet the pseudocount value
   */
  public void setPseudoCountDirichlet(double pseudoCountDirichlet) {
    m_pseudoCountDirichlet = pseudoCountDirichlet;
  } 
   
  /** Get the pseudo-count value for Dirichlet smoothing
   * @return the pseudocount value
   */
  public double getPseudoCountDirichlet() {
    return m_pseudoCountDirichlet;
  } 

  /** Set the lambda parameter for Jelinek-Mercer smoothing
   * @param lambda
   */
  public void setLambdaJM(double lambdaJM) {
    m_lambdaJM = lambdaJM;
  }

  /** Get the lambda parameter for Jelinek-Mercer smoothing
   * @return lambda
   */
  public double getLambdaJM() {
    return m_lambdaJM;
  }

    /** The computation of a metric can be either based on distance, or on similarity
   * @returns true because euclidean metrict fundamentally computes distance
   */
  public boolean isDistanceBased() {
    return true;
  }

  /** Switch between regular KL divergence and I-divergence */
  public void setUseIDivergence(boolean useID) {
    m_useIDivergence = useID;
  } 

  /** Check whether regular KL divergence or I-divergence is used */
  public boolean getUseIDivergence() {
    return m_useIDivergence;
  } 
  
  /**
   * Given a cluster of instances, return the centroid of that cluster
   * @param instances objects belonging to a cluster
   * @param fastMode whether fast mode should be used for SparseInstances
   * @param normalized normalize centroids for SPKMeans
   * @return a centroid instance for the given cluster
   */
  public Instance getCentroidInstance(Instances instances, boolean fastMode, boolean normalized) {
    double [] values = new double[instances.numAttributes()];
    if (fastMode) {
      values = meanOrMode(instances); // uses fast meanOrMode
    } else {
      for (int j = 0; j < instances.numAttributes(); j++) {
	values[j] = instances.meanOrMode(j); // uses usual meanOrMode
      }
    }
    Instance centroid = new Instance(1.0, values);
    // cluster centroids are dense in SPKMeans
    if (normalized) {
      try {
	normalizeInstanceWeighted(centroid);
      } catch (Exception e) {
	e.printStackTrace();
      }
    }
    return centroid;
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

    if (Utils.getFlag('R', options)) {
      setTrainable(Utils.getFlag('R', options));
      setExternal(Utils.getFlag('X', options));
      String metricLearnerString = Utils.getOption('L', options);
      if (metricLearnerString.length() != 0) {
	String [] metricLearnerSpec = Utils.splitOptions(metricLearnerString);
	String metricLearnerName = metricLearnerSpec[0];
	metricLearnerSpec[0] = "";
	System.out.println("Got metric learner " + metricLearnerName + " spec: " + metricLearnerSpec);
	setMetricLearner(MetricLearner.forName(metricLearnerName, metricLearnerSpec));
      } 
    }      
    

    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the classifier specification string, which contains the class name of
   * the classifier and any options to the classifier
   *
   * @return the classifier string.
   */
  protected String getMetricLearnerSpec() {
    if (m_metricLearner instanceof OptionHandler) {
      return m_metricLearner.getClass().getName() + " "
	+ Utils.joinOptions(((OptionHandler)m_metricLearner).getOptions());
    }
    return m_metricLearner.getClass().getName();
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
   * Gets the current settings of KLP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [55];
    int current = 0;

    if (m_useIDivergence) {
      options[current++] = "-I";
    }

    if (m_conversionType == CONVERSION_EXPONENTIAL) {
      options[current++] = "-E";
    } else if (m_conversionType == CONVERSION_UNIT) {
      options[current++] = "-U";
    }

    if (m_smoothingType == SMOOTHING_DIRICHLET) {
      options[current++] = "-D";
      options[current++] = "" + m_pseudoCountDirichlet;
    } else if (m_smoothingType == SMOOTHING_JELINEK_MERCER) {
      options[current++] = "-J";
      options[current++] = "" + m_lambdaJM;
    }

    if (m_useSmoothing) {
      options[current++] = "-S";
      options[current++] = "" + m_alpha;
      options[current++] = "-R";
      options[current++] = "" + m_alphaDecayRate;
    } 
    
    if (m_trainable) {
      options[current++] = "-R";
      if (m_external) {
	options[current++] = "-X";
      }

      options[current++] = "-L";
      options[current++] = Utils.removeSubstring(m_metricLearner.getClass().getName(), "weka.core.metrics.");
      if (m_metricLearner instanceof OptionHandler) {
	String[] metricOptions = ((OptionHandler)m_metricLearner).getOptions();
	for (int i = 0; i < metricOptions.length; i++) {
	  options[current++] = metricOptions[i];
	}
      }
    }
	
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /** Create a copy of this metric */
  public Object clone() {
    KL m = null; 
    m = (KL) super.clone();
    
    // clone the fields
    // for now clone a metric learner via serialization; TODO:  implement proper cloning in MetricLearners
    System.out.println("New alpha=" + m_alpha);
    try { 
      SerializedObject so = new SerializedObject(m_metricLearner);
      m.m_metricLearner = (MetricLearner) so.getObject();
    } catch (Exception e) {
      System.err.println("Problems cloning m_metricLearner while cloning KL");
    }
    return m;
  }

    
  public static void main(String[] args) {
    try {
//        // Create numeric attributes 
//        Attribute attr1 = new Attribute("attr1");
//        Attribute attr2 = new Attribute("attr2");
//        Attribute attr3 = new Attribute("attr3");
//        Attribute attr4 = new Attribute("attr4");
      
//        // Create vector to hold nominal values "first", "second", "third" 
//        FastVector my_nominal_values = new FastVector(3); 
//        my_nominal_values.addElement("first"); 
//        my_nominal_values.addElement("second"); 
//        my_nominal_values.addElement("third"); 
      
//        // Create nominal attribute "classAttr" 
//        Attribute classAttr = new Attribute("classAttr", my_nominal_values);
      
//        // Create vector of the above attributes 
//        FastVector attributes = new FastVector(4);
//        attributes.addElement(attr1);
//        attributes.addElement(attr2);
//        attributes.addElement(attr3);
//        attributes.addElement(attr4);
//        attributes.addElement(classAttr);
      
//        // Create the empty dataset with above attributes
//        Instances dataset = new Instances("dataset", attributes, 0);
      
//        // Make position the class attribute
//        dataset.setClassIndex(classAttr.index());
      
//        // Create a sparse instance with three attribute values
//        SparseInstance s_inst1 = new SparseInstance(1, new double[0], new int[0], 4);
//        s_inst1.setValue(attr1, 0.5);
//        s_inst1.setValue(attr3, 0.5);
//        s_inst1.setValue(classAttr, "third");

//        // Create a sparse instance with three attribute values
//        SparseInstance s_inst2 = new SparseInstance(1, new double[0], new int[0], 4);
//        s_inst2.setValue(attr2, 0.5);
//        s_inst2.setValue(attr3, 0.5);
//        s_inst2.setValue(classAttr,"second");

//        // Create a non-sparse instance with all attribute values
//        Instance inst1 = new Instance(5);
//        inst1.setValue(attr1, 3);
//        inst1.setValue(attr2, 4);
//        inst1.setValue(attr3, 5);
//        inst1.setValue(attr4, 2);
//        inst1.setValue(classAttr, "first");

//        // Create a sparse instance with three attribute values
//        Instance inst2 = new Instance(5);
//        inst2.setValue(attr1, 2);
//        inst2.setValue(attr2, 2);
//        inst2.setValue(attr3, 2);
//        inst2.setValue(attr4, 3);
//        inst2.setValue(classAttr, "second");
//        // Set instances' dataset to be the dataset "dataset"
//        s_inst1.setDataset(dataset);
//        s_inst2.setDataset(dataset);
//        inst1.setDataset(dataset);
//        inst2.setDataset(dataset);
      
//        // Print the instances
//        System.out.println("Sparse instance S1: " + s_inst1);
//        System.out.println("Sparse instance S2: " + s_inst2);
//        System.out.println("Non-sparse instance NS1: " + inst1);
//        System.out.println("Non-sparse instance NS2: " + inst2);
      
//        // Print the class attribute
//        System.out.println("Class attribute: " + s_inst1.classAttribute());
      
//        // Print the class index
//        System.out.println("Class index: " + s_inst1.classIndex());

//        // Create a new metric and print the distances
//        KL metric = new KL(4);
//        metric.setClassIndex(classAttr.index());
//        System.out.println("Distance between S1 and S2: " + metric.distanceJS(s_inst1, s_inst2));
//        System.out.println("Distance between S1 and NS1: " + metric.distanceJS(s_inst1, inst1));
//        System.out.println("Distance between NS1 and S1: " + metric.distanceJS(inst1, s_inst1));
//        System.out.println("Distance between NS1 and NS2: " + metric.distanceJS(inst1, inst2));
//        System.out.println("\nDistance-similarity conversion type: " +
//  			 metric.getConversionType().getSelectedTag().getReadable());
//        System.out.println("Similarity between S1 and S2: " + metric.similarity(s_inst1, s_inst2));
//        System.out.println("Similarity between S1 and NS1: " + metric.similarity(s_inst1, inst1));
//        System.out.println("Similarity between NS1 and S1: " + metric.similarity(inst1, s_inst1));
//        System.out.println("Similarity between NS1 and NS2: " + metric.similarity(inst1, inst2));
//        System.out.println();
//        System.out.println("Difference instance S1-S2: " + metric.createDiffInstance(s_inst1, s_inst2));
//        System.out.println("Difference instance S1-NS1: " + metric.createDiffInstance(s_inst1, inst1));
//        System.out.println("Difference instance NS1-S1: " + metric.createDiffInstance(inst1, s_inst1));
//        System.out.println("Difference instance NS1-NS2: " + metric.createDiffInstance(inst1, inst2));



      Instances instances = new Instances(new FileReader ("/tmp/INST.arff"));

      KL metric = new KL();
      metric.buildMetric(instances);

      Instance newCentroid = instances.instance(0);
      Instance oldCentroid = instances.instance(1);
      instances.delete(0);
      instances.delete(0);

      metric.setSmoothingType(new SelectedTag(SMOOTHING_DIRICHLET, TAGS_SMOOTHING));
      metric.setUseIDivergence(true);

      double [] values = new double[instances.numAttributes()]; 

      for (int i = 0; i < instances.numInstances(); i++) {
	Instance instance = instances.instance(i);
	for (int j = 0; j < instance.numAttributes(); j++) {
	  values[j] += instance.value(j); 
	}
      }

      for (int j = 0; j < instances.numAttributes(); j++) {
	values[j] /= instances.numInstances();
      }
      Instance saneCentroid = new SparseInstance(1.0, values);

      
      System.out.println("NumInstances=" + instances.numInstances());
      double prevTotal=0, currTotal=0, saneTotal=0;
      for (int i = 0; i < instances.numInstances(); i++) {
	double prevPen = metric.distanceNonSparse(instances.instance(i), oldCentroid);
	double currPen = metric.distanceNonSparse(instances.instance(i), newCentroid);
	double sanePen = metric.distanceNonSparse(instances.instance(i), saneCentroid);
	prevTotal += prevPen;
	currTotal += currPen;
	saneTotal += sanePen; 
	System.out.println(prevPen + " -> " + currPen + "\t" + sanePen);
      }
      System.out.println("====\n" + prevTotal + "\t" + currTotal + "\t" + saneTotal); 
      
      
    } catch (Exception e) {
      e.printStackTrace();
    }	
  } 
}

