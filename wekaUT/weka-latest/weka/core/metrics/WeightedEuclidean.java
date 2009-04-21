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
 *    WeightedEuclidean.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import weka.core.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;

/** 
 * WeightedEuclidean class
 *
 * Implements weighted euclidean distance metric
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.13 $
 */

public class WeightedEuclidean extends  LearnableMetric implements OptionHandler {

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
  
  /**
   * Create a new metric.
   * @param numAttributes the number of attributes that the metric will work on
   */ 
  public WeightedEuclidean(int numAttributes) throws Exception {
    buildMetric(numAttributes);
  }

  /** Create a default new metric */
  public WeightedEuclidean() {
  } 
   
  /**
   * Creates a new metric which takes specified attributes.
   *
   * @param _attrIdxs An array containing attribute indeces that will
   * be used in the metric
   */
  public WeightedEuclidean(int[] _attrIdxs) throws Exception {
    setAttrIdxs(_attrIdxs);
    buildMetric(_attrIdxs.length);	
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
    // either pass the computation to the external classifier, or do the work yourself
    if (m_trainable && m_external && m_trained) {
      return m_metricLearner.getDistance(instance1, instance2);
    } else {
      return distanceInternal(instance1, instance2);
    }
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


  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceInternal(Instance instance1, Instance instance2) throws Exception {
    if (instance1 instanceof SparseInstance && instance2 instanceof SparseInstance) {
      return distanceSparse((SparseInstance)instance1, (SparseInstance)instance2);
    } else if (instance1 instanceof SparseInstance) {
      return distanceSparseNonSparse((SparseInstance)instance1, instance2);
    }  else if (instance2 instanceof SparseInstance) {
      return distanceSparseNonSparse((SparseInstance)instance2, instance1);
    } else {
      return distanceNonSparse(instance1, instance2);
    }
  }
    

  /** Returns a distance value between two sparse instances. 
   * @param instance1 First sparse instance.
   * @param instance2 Second sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceSparse(SparseInstance instance1, SparseInstance instance2) throws Exception {
    double distance = 0, value1, value2;
    SparseInstance inst1 = (SparseInstance) instance1;
    SparseInstance inst2 = (SparseInstance) instance2;
	    
    // iterate through the attributes that are present in the first instance
    for (int i = 0; i < instance1.numValues(); i++) {
      Attribute attribute = instance1.attributeSparse(i);
      int attrIdx = attribute.index();

      // Skip the class index
      if (attrIdx == m_classIndex) {
	continue;
      } 
      value1 = instance1.value(attrIdx);

      // get the corresponding value of the second instance
      int idx2 = instance2.locateIndex(attrIdx);
      if (idx2 >=0 && attrIdx == instance2.index(idx2)) {
	value2 = instance2.value(attrIdx);
      } else {
	value2 = 0;
      }
      distance += m_attrWeights[attrIdx]  * (value2 - value1) * (value2 - value1);
    } 

    // Go through the attributes that are present in the second instance, but not first instance
    for (int i = 0; i < instance2.numValues(); i++) {
      Attribute attribute = instance2.attributeSparse(i);
      int attrIdx = attribute.index();
      // Skip the class index
      if (attrIdx == m_classIndex) {
	continue;
      }
      // only include attributes that are not present in first instance
      int idx1 = instance1.locateIndex(attrIdx);
      if (idx1 < 0 || attrIdx != instance1.index(idx1)) {
	value2 = instance2.value(attrIdx);
	distance += m_attrWeights[attrIdx]  * value2 * value2;
      } 
    }
    distance = Math.sqrt(distance);
    return distance;
  }

  /** Returns a distance value between a non-sparse instance and a sparse instance
   * @param instance1 sparse instance.
   * @param instance2 sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceSparseNonSparse(SparseInstance instance1, Instance instance2) throws Exception {
    double diff, distance = 0;
    double [] values2 = instance2.toDoubleArray();

    for (int i = 0; i < values2.length; i++) {
      // Skip the class index
      if (i == m_classIndex) {
	continue;
      }
      diff = values2[i] - instance1.value(i);
      distance += m_attrWeights[i]  * diff * diff;
    }
    distance = Math.sqrt(distance);
    return distance;
  };


  /** Returns a distance value between non-sparse instances without using the weights
   * @param instance1 non-sparse instance.
   * @param instance2 non-sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceNonSparse(Instance instance1, Instance instance2) throws Exception {
    double value1, value2, diff, distance = 0;
    double [] values1 = instance1.toDoubleArray();
    double [] values2 = instance2.toDoubleArray();
    // Go through all attributes
    for (int i = 0; i < values1.length; i++) {
      // Skip the class index
      if (i == m_classIndex) {
	continue;
      } 
      diff = values1[i] - values2[i];
      distance += m_attrWeights[i] * diff * diff;
    }
    distance = Math.sqrt(distance);
    return distance;
  };


  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceNonWeighted(Instance instance1, Instance instance2) throws Exception {
    if (instance1 instanceof SparseInstance && instance2 instanceof SparseInstance) {
      return distanceSparseNonWeighted((SparseInstance)instance1, (SparseInstance)instance2);
    } else if (instance1 instanceof SparseInstance) {
      return distanceSparseNonSparseNonWeighted((SparseInstance)instance1, instance2);
    }  else if (instance2 instanceof SparseInstance) {
      return distanceSparseNonSparseNonWeighted((SparseInstance)instance2, instance1);
    } else {
      return distanceNonSparseNonWeighted(instance1, instance2);
    }
  }
    

  /** Returns a distance value between two sparse instances without using the weights. 
   * @param instance1 First sparse instance.
   * @param instance2 Second sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceSparseNonWeighted(SparseInstance instance1, SparseInstance instance2) throws Exception {
    double distance = 0, value1, value2;
    SparseInstance inst1 = (SparseInstance) instance1;
    SparseInstance inst2 = (SparseInstance) instance2;
	    
    // iterate through the attributes that are present in the first instance
    for (int i = 0; i < instance1.numValues(); i++) {
      Attribute attribute = instance1.attributeSparse(i);
      int attrIdx = attribute.index();

      // Skip the class index
      if (attrIdx == m_classIndex) {
	continue;
      } 
      value1 = instance1.value(attrIdx);

      // get the corresponding value of the second instance
      int idx2 = instance2.locateIndex(attrIdx);
      if (idx2 >=0 && attrIdx == instance2.index(idx2)) {
	value2 = instance2.value(attrIdx);
      } else {
	value2 = 0;
      }
      distance += (value2 - value1) * (value2 - value1);
    } 

    // Go through the attributes that are present in the second instance, but not first instance
    for (int i = 0; i < instance2.numValues(); i++) {
      Attribute attribute = instance2.attributeSparse(i);
      int attrIdx = attribute.index();
      // Skip the class index
      if (attrIdx == m_classIndex) {
	continue;
      }
      // only include attributes that are not present in first instance
      int idx1 = instance1.locateIndex(attrIdx);
      if (idx1 < 0 || attrIdx != instance1.index(idx1)) {
	value2 = instance2.value(attrIdx);
	distance += value2 * value2;
      } 
    }
    distance = Math.sqrt(distance);
    return distance;
  }

  /** Returns a distance value between a non-sparse instance and a sparse instance
   * @param instance1 sparse instance.
   * @param instance2 sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceSparseNonSparseNonWeighted(SparseInstance instance1, Instance instance2) throws Exception {
    double diff, distance = 0;
    double [] values2 = instance2.toDoubleArray();

    for (int i = 0; i < values2.length; i++) {
      // Skip the class index
      if (i == m_classIndex) {
	continue;
      }
      diff = values2[i] - instance1.value(i);
      distance += diff * diff;
    }
    distance = Math.sqrt(distance);
    return distance;
  };


  /** Returns a distance value between non-sparse instances (or a non-sparse instance and a sparse instance)
   * without using the weights
   * @param instance1 non-sparse instance.
   * @param instance2 non-sparse instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceNonSparseNonWeighted(Instance instance1, Instance instance2) throws Exception {
    double value1, value2, diff, distance = 0;
    double [] values1 = instance1.toDoubleArray();
    double [] values2 = instance2.toDoubleArray();
    // Go through all attributes
    for (int i = 0; i < values1.length; i++) {
      // Skip the class index
      if (i == m_classIndex) {
	continue;
      } 
      diff = values1[i] - values2[i];
      distance += diff * diff;
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
      gradients[i] = 1.0 / (2*distance) * (val2 - val1) * (val2 - val1);
    }
    return gradients;
  }
  

  /** Train the metric
   */
  public void learnMetric (Instances data) throws Exception {
    if (m_metricLearner == null) {
      System.err.println("Metric learner for WeightedEuclidean is not initalized. No training was conducted");
      return;
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
    int maxNumValues = instance1.dataset().numAttributes();  // the overall number of attributes
    int classIndex = instance1.classIndex();
	
    // arrays that will hold values and internal indeces of attribute indices
    // these will be cut off later
    double [] attrValues = new double[maxNumValues];
    int [] indices = new int[maxNumValues];
    int counter = 0;
    Arrays.fill(attrValues, Double.NaN);
    Arrays.fill(indices, Integer.MAX_VALUE);

    // iterate through the attributes that are present
    for (int i = 0; i < maxNumValues; i++) {
      if (i != classIndex) {  // skip class attributes
	int idx1 = instance1.locateIndex(i); 
	int idx2 = instance2.locateIndex(i);
	if ((idx1 >=0 && i == instance1.index(idx1)) || (idx2 >=0 && i == instance2.index(idx2))) {
	  attrValues[counter] = (float) ((instance1.value(i) - instance2.value(i)) * (instance1.value(i) - instance2.value(i)));
	  indices[counter] = i;
	  counter++;
	} 
      } else { // add the class value
	if (instance1.classValue() == instance2.classValue()) {
	  attrValues[counter] = 1;
	} else {
	  attrValues[counter] = -1;
	}
	indices[counter] = instance1.classIndex();
	counter++;
      }
    }
    

    // Create the sparse difference instance
    double [] trueAttrValues = new double[counter];
    int [] trueIndices = new int[counter];
    for (int i = 0; i < counter; i++) {
      trueAttrValues[i] = attrValues[i];
      trueIndices[i] = indices[i];
    }
    SparseInstance diffInstance = new SparseInstance(1.0, trueAttrValues, trueIndices, maxNumValues);
    diffInstance.setDataset(instance1.dataset());
    return diffInstance;
  }

  /**
   * Create an instance with features corresponding to dot-product components of the two given instances
   * @param instance1 first sparse instance
   * @param instance2 second non-sparse instance
   */
  protected Instance createDiffInstanceSparseNonSparse (SparseInstance instance1, Instance instance2) {
    double[] values2 = instance2.toDoubleArray();
    int numAttributes = values2.length;
    // create an extra attribute if there was no class index originally
    int classIndex = instance1.classIndex();
    if (classIndex < 0) {
      classIndex = numAttributes;
      numAttributes++;
    }
    double[] diffInstanceValues = new double[numAttributes];  

    // iterate through the attributes that are present in the sparse instance
    for (int i = 0; i < numAttributes; i++) {
      if (i != classIndex) {  // round up to float significance to be able to weed out duplicates later
	diffInstanceValues[i] = (float) ((instance1.value(i) - values2[i]) * (instance1.value(i) - values2[i]));
      } else {  // class value
	if (instance1.value(i) == values2[i]) {
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
	diffInstanceValues[i] = (float) ((values1[i] - values2[i]) * (values1[i] - values2[i]));
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
   * Gets the current settings of WeightedEuclideanP.
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
    WeightedEuclidean m = null; 
    m = (WeightedEuclidean) super.clone();
    
    // clone the fields
    // for now clone a metric learner via serialization; TODO:  implement proper cloning in MetricLearners
    try { 
      SerializedObject so = new SerializedObject(m_metricLearner);
      m.m_metricLearner = (MetricLearner) so.getObject();
    } catch (Exception e) {
      System.err.println("Problems cloning m_metricLearner while cloning WeightedEuclidean");
    }
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
      WeightedEuclidean metric = new WeightedEuclidean(4);
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

