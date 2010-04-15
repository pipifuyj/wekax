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
 *    SumInstanceMetric.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.deduping.metrics;

import java.util.*;
import java.text.SimpleDateFormat;
import java.io.*;

import weka.deduping.*;
import weka.core.*;

import weka.classifiers.DistributionClassifier;
import weka.classifiers.functions.SMO;

/** 
 * SumInstanceMetric class simply adds
 * values returned by StringMetrics on individual fields
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.5 $
 */

public class SumInstanceMetric extends InstanceMetric implements OptionHandler, Serializable {
  
  /** A selector object that will create training sets */
  PairwiseSelector m_selector = new PairwiseSelector();

  /** An array of StringMetrics that are to be used on each attribute */
  /*protected*/public StringMetric [] m_stringMetrics = null; 
  protected StringMetric m_metric = new AffineMetric();

  /** The number of positive pairs desired for training */
  protected int m_numPosPairs = 500;
  protected int m_numNegPairs = 500;


  /** We may require objects to have a minimum number of
   * common tokens for them to be considered
   * for distance computation
   */
  protected int m_minCommonTokens = 0; 

  
  /** A default constructor */
  public SumInstanceMetric() {
  } 
      

  /**
   * Generates a new SumInstanceMetric based on specified 
   * attributes. Has to initialize all fields of the metric with
   * default values.
   *
   * @param numAttributes the number of attributes that the metric will work on
   * @exception Exception if the distance metric has not been
   * generated successfully.  */
  public void buildInstanceMetric(int[] attrIdxs) throws Exception {
    // initialize the array of metrics for each attribute
    m_attrIdxs = attrIdxs;
    m_stringMetrics = new StringMetric[m_attrIdxs.length];
    for (int i = 0; i < m_stringMetrics.length; i++) {
      m_stringMetrics[i] = (StringMetric) m_metric.clone();
    } 
  }

  /**
   * Create a new metric for operating on specified instances
   * @param data instances that the metric will be used on
   */
  public void trainInstanceMetric(Instances trainData, Instances testData) throws Exception {
    m_selector.initSelector(trainData);
    
    // if we have data-dependent metrics (e.g. vector-space), build them with available data
    if (m_metric instanceof DataDependentStringMetric) {
      for (int i = 0; i < m_stringMetrics.length; i++) {
	ArrayList stringList = getStringList(trainData, testData, m_attrIdxs[i]);
	((DataDependentStringMetric)m_stringMetrics[i]).buildMetric(stringList);
      }
    }

    // train all the learnable metrics
    if (m_metric instanceof LearnableStringMetric) {    
      for (int i = 0; i < m_stringMetrics.length; i++) {
	ArrayList strPairList = m_selector.getStringPairList(trainData, m_attrIdxs[i],
							     m_numPosPairs, m_numNegPairs, m_stringMetrics[i]);
	m_numActualPosPairs = m_numPosPairs;
	m_numActualNegPairs = m_numNegPairs;
	// begin: creating transductive pairs for metric learning
	for (int j = 0; j < 00; j++) {
	  Random r = new Random(j);
	  int idx1, idx2;
	  idx1 = r.nextInt(testData.numInstances());
	  do {
	    idx2 = r.nextInt(testData.numInstances());
	  } while (idx2 == idx1);
	  
	  StringPair pair = new StringPair(testData.instance(idx1).stringValue(m_attrIdxs[i]),
					   testData.instance(idx2).stringValue(m_attrIdxs[i]),
					   true, 1);
	  strPairList.add(pair);	  
	}
	// end:  creating  transductive pairs for metric learning
	
	((LearnableStringMetric)m_stringMetrics[i]).trainMetric(strPairList);
      }
    }
    System.out.println(getTimestamp() + " Created a SumInstanceMetric.");
  }

  /** An internal method for creating a list of strings for a particular attribute
   * from two sets of instances:  trianing and test data
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
   * Returns distance between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double distance(Instance instance1, Instance instance2) throws Exception {
    // go through all metrics collecting the values of distances
    double distance = 0;
    for (int i = 0; i < m_stringMetrics.length; i++) {
      String str1 = instance1.stringValue(m_attrIdxs[i]);
      String str2 = instance2.stringValue(m_attrIdxs[i]);

      if (m_minCommonTokens > 0) {
	if (numCommonTokens(str1, str2) >= m_minCommonTokens) {
	  double d = m_stringMetrics[i].distance(str1, str2);
	  distance += d;
	} else {   // there are too few common tokens; skip 
	  distance = Double.MAX_VALUE;
	} 
      } else {  // minCommonTokens = 0; we always compute distance
	double d = m_stringMetrics[i].distance(str1, str2);
	distance += d;
      }
    }
    
    return distance;
  } 

  /**
   * Returns similarity between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarity(Instance instance1, Instance instance2) throws Exception {
    // go through all metrics collecting the values of distances
    double similarity = 0;
    for (int i = 0; i < m_stringMetrics.length; i++) {
      String str1 = instance1.stringValue(m_attrIdxs[i]);
      String str2 = instance2.stringValue(m_attrIdxs[i]);
      similarity += m_stringMetrics[i].similarity(str1, str2);
    }
    return similarity;
  } 

  /** The computation of a metric can be either based on distance, or on similarity
   * @returns true if the underlying metric computes distance, false if similarity
   */
  public boolean isDistanceBased() {
    return true;
  };

  /**
   * Set the baseline metric
   *
   * @param metric the string metric to be used as the baseline on each string attribute
   */
  public void setMetric (StringMetric metric) {
    m_metric = metric;
  }

  /**
   * Get the baseline metric
   *
   * @returns the baseline metric for each attribute
   */
  public StringMetric getMetric () {
    return m_metric;
  }

  /** Set the pairwise selector for this metric
   * @param selector a new pairwise selector
   */
  public void setSelector(PairwiseSelector selector) {
    m_selector = selector;
  }

  /** Get the pairwise selector for this metric
   * @param selector a new pairwise selector
   */
  public PairwiseSelector getSelector() {
    return m_selector;
  } 
  


  /** Set the number of same-class training pairs
   * @param numPosPairs the number of same-class training pairs to create for training
   */
  public void setNumPosPairs(int numPosPairs) {
    m_numPosPairs = numPosPairs;
  }

  /** Get  the number of same-class training pairs
   * @return the number of same-class training pairs to create for training
   */
  public int getNumPosPairs() {
    return m_numPosPairs;
  } 

  /** Set the number of different-class training pairs
   * @param numNegPairs the number of different-class training pairs to create for training
   */
  public void setNumNegPairs(int numNegPairs) {
    m_numNegPairs = numNegPairs;
  }

  /** Get  the number of different-class training pairs
   * @return the number of different-class training pairs to create for training
   */
  public int getNumNegPairs() {
    return m_numNegPairs;
  }

  /** Set the minimum number of common tokens that is required from objects
   * to be considered for distance computation
   * @param minCommonTokens the minimum number of tokens in common that is required
   * from objects to be considered for distance computation
   */
  public void setMinCommonTokens(int minCommonTokens) {
    m_minCommonTokens = minCommonTokens;
  }


  /** Get the minimum number of common tokens that is required from objects
   * to be considered for distance computation
   * @return the minimum number of tokens in common that is required
   * from objects to be considered for distance computation
   */
  public int getMinCommonTokens() {
    return m_minCommonTokens;
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


  /** return the number of tokens that two strings have in commmon */
  public static int numCommonTokens(String s1, String s2) {
    String delimiters = " \t\n\r\f\'\"\\!@#$%^&*()    ";
    HashSet set1 = new HashSet(); 
    StringTokenizer tokenizer = new StringTokenizer(s1, delimiters);
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      set1.add(token);
    }

    int numCommon = 0; 
    tokenizer = new StringTokenizer(s2, delimiters);
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      if (set1.contains(token)) {
	numCommon++;
      }
    }
    return numCommon;
  } 

  
      /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   **/
  public Enumeration listOptions() {
    Vector newVector = new Vector(0);
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

    System.err.println("TODO!  this method has not been implemented properly");
    String metricString = Utils.getOption('M', options);
    if (metricString.length() != 0) {
      String[] metricSpec = Utils.splitOptions(metricString);
      String metricName = metricSpec[0]; 
      metricSpec[0] = "";
      System.out.println("Metric name: " + metricName + "\nMetric parameters: " + concatStringArray(metricSpec));
      setMetric(StringMetric.forName(metricName, metricSpec));
    }

  }


  /**
   * Gets the current settings of Greedy Agglomerative Clustering
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [60];
    int current = 0;

    if (m_minCommonTokens > 0) {
      options[current++] = "-t";
      options[current++] = "" + m_minCommonTokens;
    }

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

    options[current++] = "-M";
    options[current++] = Utils.removeSubstring(m_metric.getClass().getName(), "weka.deduping.metrics.");
    if (m_metric instanceof OptionHandler) {
      String[] metricOptions = ((OptionHandler)m_metric).getOptions();
      for (int i = 0; i < metricOptions.length; i++) {
	options[current++] = metricOptions[i];
      }
    } 

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }

}






