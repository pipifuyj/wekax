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
 *    ClassifierInstanceMetric.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.deduping.metrics;

import java.util.ArrayList;
import java.util.Vector;
import java.util.Enumeration;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.io.*;

import weka.deduping.*;
import weka.core.*;

import weka.classifiers.DistributionClassifier;
import weka.classifiers.sparse.SVMlight;
import weka.classifiers.Evaluation;

/** 
 * ClassifierInstanceMetric class employs a classifier that uses
 * values returned by various StringMetric's on individual fields
 * as features and outputs a confidence value that corresponds to
 * similarity between records
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.5 $
 */

public class ClassifierInstanceMetric extends InstanceMetric implements OptionHandler, Serializable {
  
  /** Classifier that is used for estimating similarity between records */
  protected DistributionClassifier m_classifier = new SVMlight();
  
  /** A selector object that will create training sets */
  PairwiseSelector m_selector = new PairwiseSelector();

  /** The desired number of training pairs */
  protected int m_numPosPairs = 200;
  protected int m_numNegPairs = 200;

  /** StringMetric prototype that are to be used on each field */
  protected StringMetric [] m_stringMetrics = new StringMetric[0];

  /** The actual array of metrics */
  protected StringMetric [][] m_fieldMetrics = null;

  /** A temporary dataset that contains diff-instances for training the classifier */
  protected Instances m_diffInstances = null;

  /** A default constructor */
  public ClassifierInstanceMetric() {
  } 
      

  /**
   * Generates a new ClassifierInstanceMetric that computes
   * similarity between records using the specified attributes. Has to
   * initialize all metric fields with default string metrics
   *
   * @param attrIdxs the indeces of attributes that the metric will use
   * @exception Exception if the distance metric has not been
   * generated successfully.  */
  public void buildInstanceMetric(int[] attrIdxs) throws Exception {
    // initialize the array of metrics for each attribute
    m_attrIdxs = attrIdxs;
    m_fieldMetrics = new StringMetric[m_stringMetrics.length][m_attrIdxs.length];

    for (int i = 0; i < m_stringMetrics.length; i++) {
      for (int j = 0; j < m_attrIdxs.length; j++) {
	m_fieldMetrics[i][j] = (StringMetric) m_stringMetrics[i].clone();
      }
    } 
  }

  
  /**
   * Create a new metric for operating on specified instances
   * @param trainData instances for training the metric
   * @param testData instances that will be used for testing
   */
  public void trainInstanceMetric(Instances trainData, Instances testData) throws Exception {
    m_selector.initSelector(trainData);
    
    // if we have data-dependent or trainable metrics
    // (e.g. vector-space or learnable ED), build them with available
    // test/train data

    ArrayList [] attrStringLists = null;
    for (int i = 0; i < m_stringMetrics.length; i++) {
      if (m_stringMetrics[i] instanceof DataDependentStringMetric) {

	// populate the list of strings for each attribute now that we need them
	if (attrStringLists == null) { 
	  attrStringLists = new ArrayList[m_attrIdxs.length];
	  for (int j = 0; j < m_attrIdxs.length; j++) {
	    attrStringLists[j] = getStringList(trainData, testData, m_attrIdxs[j]);
	  }
	}

	// initialize the data-dependent metric for each attribute
	for (int j = 0; j < m_attrIdxs.length; j++) {
	  ((DataDependentStringMetric)m_fieldMetrics[i][j]).buildMetric(attrStringLists[j]);
	}
      }

      // if the metric is learnable, train it
      if (m_stringMetrics[i] instanceof LearnableStringMetric) {    
	for (int j = 0; j < m_attrIdxs.length; j++) {
	  ArrayList strPairList = m_selector.getStringPairList(trainData, m_attrIdxs[j],
							       m_numPosPairs, m_numNegPairs,
							       m_fieldMetrics[i][j]);
	  ((LearnableStringMetric)m_fieldMetrics[i][j]).trainMetric(strPairList);
	}
      }
    }
    
    // train the classifier
    m_diffInstances = m_selector.getInstances(m_attrIdxs, m_fieldMetrics, m_numPosPairs, m_numNegPairs);

    // get the stats on actual training data
    AttributeStats classStats = m_diffInstances.attributeStats(m_diffInstances.classIndex());
    m_numActualPosPairs = classStats.nominalCounts[0];
    m_numActualNegPairs = classStats.nominalCounts[1];
    
    // SANITY CHECK - CROSS-VALIDATION
    if (false) { 
      // dump diff-instances into a temporary file
      try {
	File diffDir = new File("/tmp/diff");
	diffDir.mkdir();
	String diffName = trainData.relationName() + "." +
	  Utils.removeSubstring(m_fieldMetrics[0].getClass().getName(), "weka.deduping.metrics.");
	m_diffInstances.setRelationName(diffName);
	PrintWriter writer = new PrintWriter(new BufferedOutputStream (new FileOutputStream(diffDir.getPath() + "/" +
											    diffName + ".arff")));
	writer.println(m_diffInstances.toString());
	writer.close();

	// Do a sanity check - dump out the diffInstances, and
	// evaluation classification with an SVM. 
	long trainTimeStart = System.currentTimeMillis();
	SVMlight classifier = new SVMlight();
	Evaluation eval = new Evaluation(m_diffInstances);
	eval.crossValidateModel(classifier, m_diffInstances, 5);
	writer = new PrintWriter(new BufferedOutputStream (new FileOutputStream(diffDir.getPath() + "/" +
										diffName + ".dat", true)));
	writer.println(eval.pctCorrect());
	writer.close();
	System.out.println("** Record Sanity:" + (System.currentTimeMillis() - trainTimeStart) + " ms; " +
			   eval.pctCorrect() + "% correct\t" +
			   eval.numFalseNegatives(0) + "(" + eval.falseNegativeRate(0) + "%) false negatives\t" +
			   eval.numFalsePositives(0) + "(" + eval.falsePositiveRate(0) + "%) false positives\t");
      
      } catch (Exception e) {
	e.printStackTrace();
	System.out.println(e.toString()); 
      }
    }
    // END SANITY CHECK

    System.out.println(getTimestamp() + ":  Building " + m_classifier.getClass().getName());
    m_classifier.buildClassifier(m_diffInstances);
    System.out.println(getTimestamp() + ":  Done building " + m_classifier.getClass().getName());
  }

  
  /** An internal method for creating a list of strings for a
   * particular attribute from two sets of instances: trianing and
   * test data
   * @param trainData a dataset of records in the training fold
   * @param testData a dataset of records in the testing fold
   * @param attrIdx the index of the attribute for which strings are to be collected
   * @return a list of strings that occur for this attribute; duplicates are allowed
   */
  protected ArrayList getStringList(Instances trainData, Instances testData, int attrIdx) {
    ArrayList stringList = new ArrayList();

    // go through the training data and get all string values for that attribute
    if (trainData != null) { 
      for (int i = 0; i < trainData.numInstances(); i++) {
	Instance instance = trainData.instance(i);
	String value = instance.stringValue(attrIdx);
	stringList.add(value);
      }
    }

    // go through the test data and get all string values for that attribute
    for (int i = 0; i < testData.numInstances(); i++) {
      Instance instance = testData.instance(i);
      String value = instance.stringValue(attrIdx);
      stringList.add(value);
    }

    return stringList;
  } 

    
  /**
   * Returns distance between two records 
   * @param instance1 First record.
   * @param instance2 Second record.
   * @exception Exception if distance could not be calculated.
   */
  public double distance(Instance instance1, Instance instance2) throws Exception {
    // go through all metrics collecting the values of distances for different attributes
    double[] distances = new double[m_attrIdxs.length * m_stringMetrics.length + 1];
    int counter = 0; 
    for (int i = 0; i < m_attrIdxs.length; i++) {
      String str1 = instance1.stringValue(m_attrIdxs[i]);
      String str2 = instance2.stringValue(m_attrIdxs[i]);

      for (int j = 0; j < m_stringMetrics.length; j++) { 
	if (m_stringMetrics[j].isDistanceBased()) { 
	  distances[counter++] = m_fieldMetrics[j][i].distance(str1, str2);
	} else {
	  distances[counter++] = m_fieldMetrics[j][i].similarity(str1, str2);
	}
      }
    }
      
    Instance diffInstance = new Instance(1.0, distances);
    diffInstance.setDataset(m_diffInstances);
    return m_classifier.distributionForInstance(diffInstance)[1];
  } 

  /**
   * Returns similarity between two records
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be calculated.
   */
  public double similarity(Instance instance1, Instance instance2) throws Exception {
    double d = distance(instance1, instance2);
    return Math.exp(-d);
  } 

  /** The computation can be either based on distance, or on similarity
   * @returns true if the underlying metric computes distance, false if similarity
   */
  public boolean isDistanceBased() {
    return true;
  };

  /**
   * Set the classifier
   *
   * @param classifier the classifier
   */
  public void setClassifier (DistributionClassifier classifier) {
    m_classifier = classifier;
  }

  /**
   * Get the classifier
   *
   * @returns the classifierthat this metric employs
   */
  public DistributionClassifier getClassifier () {
    return m_classifier;
  }

  /**
   * Set the baseline metric
   *
   * @param metrics string metrics that will used on each string attribute
   */
  public void setStringMetrics (StringMetric[] metrics) {
    m_stringMetrics = metrics;
  }

  /**
   * Get the baseline string metrics
   *
   * @return the string metrics that are used for each field
   */
  public StringMetric[] getStringMetrics () {
    return m_stringMetrics;
  }


  /** Set the pairwise selector for this metric
   * @param selector a new pairwise selector
   */
  public void setSelector(PairwiseSelector selector) {
    m_selector = selector;
  }

  /** Get the pairwise selector for this metric
   * @return the pairwise selector
   */
  public PairwiseSelector getSelector() {
    return m_selector;
  } 
  
  /** Set the number of same-class training pairs that is desired
   * @param numPosPairs the number of same-class training pairs to be
   * created for training the classifier
   */
  public void setNumPosPairs(int numPosPairs) {
    m_numPosPairs = numPosPairs;
  }

  /** Get the number of same-class training pairs
   * @return the number of same-class training pairs to create for
   * training the classifier
   */
  public int getNumPosPairs() {
    return m_numPosPairs;
  } 

  /** Set the number of different-class training pairs
   * @param numNegPairs the number of different-class training pairs
   * to create for training the classifier
   */
  public void setNumNegPairs(int numNegPairs) {
    m_numNegPairs = numNegPairs;
  }

  /** Get the number of different-class training pairs
   * @return the number of different-class training pairs to create
   * for training the classifier
   */
  public int getNumNegPairs() {
    return m_numNegPairs;
  }

  /**
   * Gets a string containing current date and time.
   *
   * @return a string containing the date and time.
   */
  protected static String getTimestamp() {
    return (new SimpleDateFormat("HH:mm:ss:")).format(new Date());
  }

  /** A little helper to create a single String from an array of Strings
   * @param strings an array of strings
   * @returns a single concatenated string
   */
  public static String concatStringArray(String[] strings) {
    StringBuffer buffer = new StringBuffer();
    for (int i = 0; i < strings.length; i++) {
      buffer.append(strings[i]);
      buffer.append(" ");
    }
    return buffer.toString();
  }

  /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   **/
  public Enumeration listOptions() {
    Vector newVector = new Vector(2);
    newVector.addElement(new Option("\tMetric.\n"
				    +"\t(default=AffineMetric)", "M", 1,"-M metric_name metric_options"));
    newVector.addElement(new Option("\tClassifier.\n"
				    +"\t(default=weka.classifiers.functions.SMO)", "C", 1,"-C clasifierName classifierOptions"));
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   *
   * Valid options are:<p>
   *
   * -M metric options <p>
   * StringMetric used <p>
   *
   * -C classifier options <p>
   * Classifier used <p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   *
   **/
  public void setOptions(String[] options) throws Exception {
    String optionString;

    // TODO:  implement command-line options
//      String metricString = Utils.getOption('M', options);
//      if (metricString.length() != 0) {
//        String[] metricSpec = Utils.splitOptions(metricString);
//        String metricName = metricSpec[0]; 
//        metricSpec[0] = "";
//        System.out.println("Metric name: " + metricName + "\nMetric parameters: " + concatStringArray(metricSpec));
//        setMetric(StringMetric.forName(metricName, metricSpec));
//      }

    String classifierString = Utils.getOption('C', options);
    if (classifierString.length() == 0) {
      throw new Exception("A classifier must be specified"
			  + " with the -C option.");
    }
    String [] classifierSpec = Utils.splitOptions(classifierString);
    if (classifierSpec.length == 0) {
      throw new Exception("Invalid classifier specification string");
    }
    String classifierName = classifierSpec[0];
    classifierSpec[0] = "";
    System.out.println("Classifier name: " + classifierName + "\nClassifier parameters: " +
		       concatStringArray(classifierSpec));
    setClassifier((DistributionClassifier) DistributionClassifier.forName(classifierName, classifierSpec));
  }


  /**
   * Gets the current settings of Greedy Agglomerative Clustering
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [200];
    int current = 0;

    if (m_selector instanceof OptionHandler) {
      String[] selectorOptions = ((OptionHandler)m_selector).getOptions();
      for (int i = 0; i < selectorOptions.length; i++) {
	options[current++] = selectorOptions[i];
      }
    }
    options[current++] = "-p";
    options[current++] = "" + m_numPosPairs;
    options[current++] = "-n";
    options[current++] = "" + m_numNegPairs;

    options[current++] = "-M" + m_stringMetrics.length;
    for (int i = 0; i < m_stringMetrics.length; i++) {
      options[current++] = Utils.removeSubstring(m_stringMetrics[i].getClass().getName(), "weka.deduping.metrics.");
      if (m_stringMetrics[i] instanceof OptionHandler) {
	String[] metricOptions = ((OptionHandler)m_stringMetrics[i]).getOptions();
	for (int j = 0; j < metricOptions.length; j++) {
	  options[current++] = metricOptions[j];
	}
      }
    } 

    options[current++] = "-C";
    options[current++] = Utils.removeSubstring(m_classifier.getClass().getName(), "weka.classifiers.");
    if (m_classifier instanceof OptionHandler) {
      String[] classifierOptions = ((OptionHandler)m_classifier).getOptions();
      for (int i = 0; i < classifierOptions.length; i++) {
	options[current++] = classifierOptions[i];
      }
    }
    
    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }

}


