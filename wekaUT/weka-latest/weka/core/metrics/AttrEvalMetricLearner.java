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
 *    MetricLearner.java
 *    Copyright (C) 2002 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import java.util.*;
import java.io.Serializable;
import java.io.*;
import java.text.SimpleDateFormat;

import java.text.DecimalFormat;
import java.text.NumberFormat;


import weka.classifiers.*;
import weka.classifiers.functions.*;
import weka.core.*;
import weka.attributeSelection.*;

/** 
 * AttrEvalMetricLearner - sets the weights of a metric
 * using scores from an attribute evaluator
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.2 $
 */

public class AttrEvalMetricLearner extends MetricLearner implements Serializable, OptionHandler {

  /** The metric that the classifier was used to learn, useful for external-calculation based metrics */
  protected LearnableMetric m_metric = null;


  /** The attribute evaluator used */
  protected ASEvaluation m_evaluator = new InfoGainAttributeEval();

  /** Create a new attribute evaluator metric learner 
   * @param classifierName the name of the classifier class to be used
   */
  public AttrEvalMetricLearner() {
  } 
    
  /**
   * Train a given met7ric using given training instances
   *
   * @param metric the metric to train
   * @param instances data to train the metric on
   * @exception Exception if training has gone bad.
   */
  public void trainMetric(LearnableMetric metric, Instances instances) throws Exception {
    // If the data doesn't have a class attribute, bail
    if (instances.classIndex() < 0 || instances.numInstances() < 2) {
      metric.m_trained = false;
      System.out.println("Problem with training data");
      return;
    }

    if (metric.getExternal()) {
      throw new Exception("AttrEvalMetricLearner cannot be used as an external distance metric!");
    }

    System.out.println(getTimestamp() + "  Starting to calculate weights" );

    m_evaluator.buildEvaluator(instances);
    double[] weights = new double[metric.getNumAttributes()];
    int posWeightsCount = 0;
    int negWeightsCount = 0;
    NumberFormat decFormat = new DecimalFormat("0.000E0#");
    for (int i = 0; i < weights.length; i++) {
      weights[i] = ((AttributeEvaluator)m_evaluator).evaluateAttribute(i);
      if (weights[i] > 0) {
	posWeightsCount++;
	if (i < 100) {
	  System.out.print(decFormat.format(weights[i]) + " " + instances.attribute(i).name() + "\t\t");
	  if (posWeightsCount % 4 == 0) System.out.println();
	}
      } else if (weights[i] < 0) {
	negWeightsCount++;
      }
    }
    metric.setWeights(normalizeWeights(weights));
    System.out.println("Top components1:");
    int[] sortedIndeces = Utils.sort(weights);
    for (int i = sortedIndeces.length-1; i >  sortedIndeces.length-300 && i >=0; i--) {
      int idx = sortedIndeces[i];
      System.out.println(i + ": " + idx + ":" + instances.attribute(idx)  + "(" + weights[idx] + ")");
    }
    System.out.println(getTimestamp() + "  Learned weights: " +
		       m_evaluator.getClass().getName() + " got " + posWeightsCount + " positive and " +
		       negWeightsCount + " negative weights out of " + weights.length);
    metric.m_trained = true;
  }

  /** Normalize weights
   * @param weights an unnormalized array of weights
   * @return a normalized array of weights
   */
  protected double[] normalizeWeights(double[] weights) {
    double sum = 0;
    for (int i = 0; i < weights.length; i++) {
      sum += weights[i];
    }
    
    double [] newWeights = new double[weights.length];
    for (int i = 0; i < weights.length; i++) {
      newWeights[i] = weights[i] / sum;
    }
    return newWeights;
  }

  /**
   * Use the Classifier for an estimation of similarity
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns sim an approximate similarity obtained from the classifier
   */
  public double getSimilarity(Instance instance1, Instance instance2) throws Exception{
    throw new Exception("InfoGainMetricLearner cannot be used as an external distance metric!");
  }

  /**
   * Use the Classifier for an estimation of distance
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns  an approximate distance obtained from the classifier
   */
  public double getDistance(Instance instance1, Instance instance2) throws Exception{
    throw new Exception("InfoGainMetricLearner cannot be used as an external distance metric!");
  }


  /**
   * Set the evaluator
   */
  public void setEvaluator(ASEvaluation evaluator) throws Exception {
    if (evaluator instanceof AttributeEvaluator) {
      m_evaluator = evaluator;
    } else {
      throw new Exception("Evaluator must be a child class of AttributeEvaluator!");
    }
  }

  /**
   * Get the evaluator
   */
  public ASEvaluation getEvaluator() {
    return m_evaluator;
  }
  
  /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [20];
    int current = 0;

    options[current++] = "-E";
    options[current++] = m_evaluator.getClass().getName();
    if (m_evaluator instanceof OptionHandler) {
      String[] evaluatorOptions = ((OptionHandler)m_evaluator).getOptions();
      for (int i = 0; i < evaluatorOptions.length; i++) {
	options[current++] = evaluatorOptions[i];
      }
    } 

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -B classifierstring
   */
  public void setOptions(String[] options) throws Exception {
  }

  /**
   * Gets a string containing current date and time.
   *
   * @return a string containing the date and time.
   */
  protected static String getTimestamp() {
    return (new SimpleDateFormat("HH:mm:ss:")).format(new Date());
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
     Vector newVector = new Vector(0);

    return newVector.elements();
  }

  /** Obtain a textual description of the metriclearner
   * @return a textual description of the metric learner
   */
  public String toString() {
    return new String("InfoGainMetricLearner " + concatStringArray(getOptions()));
  } 

    /** A little helper to create a single String from an array of Strings
   * @param strings an array of strings
   * @returns a single concatenated string, separated by commas
   */
  public static String concatStringArray(String[] strings) {
    String result = new String();
    for (int i = 0; i < strings.length; i++) {
      result = result + "\"" + strings[i] + "\" ";
    }
    return result;
  } 


}









