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
 *    GDMetricLearner.java
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
 * GDMetricLearner - sets the weights of a metric
 * using gradient descent
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */

public class GDMetricLearner extends MetricLearner implements Serializable, OptionHandler {

  /** The metric that the classifier was used to learn, useful for external-calculation based metrics */
  protected LearnableMetric m_metric = null;

  /** Maximum number of iterations */
  protected int m_maxIterations = 20;

  /** The learning rate */
  protected double m_learningRate = 0.0000001;

  /** The training data */
  protected Instances m_instances = null;
  protected ArrayList m_pairList = null;
  protected int m_numPosPairs = 200;
  protected int m_numNegPairs = 200;

  /** The convergence criterion for total weight updates */
  protected double m_epsilon = 10e-5;

  /** The pairwise selector used by the metric */
  protected PairwiseSelector m_selector = new RandomPairwiseSelector();
  
  /** Create a new gradient descent metric learner 
   * @param classifierName the name of the classifier class to be used
   */
  public GDMetricLearner() {
  } 
    
  /**
   * Train a given metric using given training instances
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
      throw new Exception("GDMetricLearner cannot be used as an external distance metric!");
    }
    
    System.out.println(getTimestamp() + "  Starting to calculate weights over " + metric.getNumAttributes() +" attributes");
    m_metric = metric;
    m_instances = instances;
    m_pairList = m_selector.createPairList(m_instances, m_numPosPairs, m_numNegPairs, metric);

    int numWeights = metric.getNumAttributes();
    double[] currentWeights = new double[numWeights];
    Arrays.fill(currentWeights,1.0/numWeights);
    metric.setWeights(currentWeights);

    int iterCount = 0;
    boolean converged = false;
    while (iterCount < m_maxIterations && !converged) {
      // calculate the gradient vector
      double [] gradients = calculateGradients(currentWeights);
      double updateTotal = 0;
      
      // update the weights
      for (int i = 0; i < numWeights; i++) {
	//	System.out.println("update: " +  gradients[i]);
	double update = m_learningRate * gradients[i];
	updateTotal += Math.abs(update);
	currentWeights[i] += update;
      }
      currentWeights = normalizeWeights(currentWeights);
      metric.setWeights(currentWeights);

      // check convergence
      if (updateTotal <= m_epsilon) {
	converged = true;
      }
      iterCount++;
    }
    printTopAttributes(currentWeights, 10, iterCount);
    System.out.println(getTimestamp() + "  Gradient descent complete after " + iterCount + " iterations");
    metric.m_trained = true;
  }


  /** A helper function that calculates the current gradient value
   * @param weights the current weights vector
   * @return the values of the partial derivatives
   */
  protected double[] calculateGradients(double[] weights) throws Exception {
    double [] gradients = new double[weights.length];

    // calculate the gradients
    for (int i = 0; i < m_pairList.size(); i++) {
      TrainingPair pair = (TrainingPair) m_pairList.get(i);

      double[] pairGradients = m_metric.getGradients(pair.instance1, pair.instance2);
      //      System.out.println(pair.instance1 + "\t" + pair.instance2 + "\t" + pair.positive);
      for (int j = 0; j < gradients.length; j++) {
	//System.out.print(gradients[j] + "(" + pairGradients[j] + ")\t");
	gradients[j] = gradients[j] + (pair.positive ? pairGradients[j] : -pairGradients[j]);
      }
      //      System.out.println();
    }
    return gradients;
  }

    /** Normalize weights
   * @param weights an unnormalized array of weights
   * @return a normalized array of weights
   */
  protected double[] normalizeWeights(double[] weights) {
    double sum = 0;
    for (int i = 0; i < weights.length; i++) {
      if (weights[i] < 0) {
	weights[i] = 0;
      } else {
	sum += weights[i];
      }
    }
    
    double [] newWeights = new double[weights.length];
    for (int i = 0; i < weights.length; i++) {
      newWeights[i] = weights[i] / sum;
    }
    return newWeights;
  }

  /** Get the norm-2 length of an instance assuming all attributes are numeric
   * and utilizing the attribute weights
   * @returns norm-2 length of an instance
   */
  public double lengthWeighted(Instance instance, double[] weights) {
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
  	  length += weights[i] * value * value;
	}
      }
    } else {  // non-sparse instance
      double[] values = instance.toDoubleArray(); 
      for (int i = 0; i < values.length; i++) {
	if (i != classIndex) {
  	  length += weights[i] * values[i] * values[i];
	}
      }
    }
    return Math.sqrt(length);
  }

  /**
   * Use the Classifier for an estimation of similarity
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns sim an approximate similarity obtained from the classifier
   */
  public double getSimilarity(Instance instance1, Instance instance2) throws Exception{
    throw new Exception("GDMetricLearner cannot be used as an external distance metric!");
  }

  /**
   * Use the Classifier for an estimation of distance
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns  an approximate distance obtained from the classifier
   */
  public double getDistance(Instance instance1, Instance instance2) throws Exception{
    throw new Exception("GDMetricLearner cannot be used as an external distance metric!");
  }

  /** Set the convergence criterion
   * @param epsilon the maximum sum of weight updates required for GD to converge
   */
  public void setEpsilon(double epsilon) {
    m_epsilon = epsilon;
  }

  /** Get the convergence criterion
   * @return the maximum sum of weight updates required for GD to converge
   */
  public double getEpsilon() {
    return m_epsilon;
  }

  /** Set the learning rate
   * @param learningRate the gradient update coefficient
   */
  public void setLearningRate(double learningRate) {
    m_learningRate = learningRate;
  }

  /** Get the learning rate
   * @return the gradient update coefficient
   */
  public double getLearningRate() {
    return m_learningRate;
  }

  /** Set the maximum number of update iterations rate
   * @param maxIterations the maximum number of gradient updates
   */
  public void setMaxIterations(int maxIterations) {
    m_maxIterations = maxIterations;
  }

  /** Get the maximum number of update iterations rate
   * @return the maximum number of gradient updates
   */
  public int getMaxIterations() {
    return m_maxIterations;
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


  /** Set the pairwise selector
   * @param selector the selector for training pairs
   */
  public void setSelector (PairwiseSelector selector) {
    m_selector = selector;
  } 

  
  /** Get the pairwise selector
   * @return the selector for training pairs
   */
  public PairwiseSelector getSelector() {
    return m_selector;
  } 
  
  /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [25];
    int current = 0;

    options[current++] = "-e";
    options[current++] = "" + m_epsilon;
    options[current++] = "-p";
    options[current++] = "" + m_numPosPairs;
    options[current++] = "-n";
    options[current++] = "" + m_numNegPairs;
    options[current++] = "-i";
    options[current++] = "" + m_maxIterations;
    options[current++] = "-l";
    options[current++] = "" + m_learningRate;
    options[current++] = "-S";
    options[current++] = m_selector.getClass().getName();


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
    return new String("GDMetricLearner " + concatStringArray(getOptions()));
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

   /** Create a lists of pairs of two kinds: pairs of instances belonging to same class,
   * and pairs of instances belonging to different classes.
   *
   */
  protected ArrayList createPairList(Instances instances, int numPosPairs, int numNegPairs) {
    ArrayList pairList = new ArrayList();

    // A hashmap where each class will be mapped to a list of instnaces belonging to it
    HashMap classInstanceMap = new HashMap();

    // A list of classes, each element is the double value of the class attribute
    ArrayList classValueList = new ArrayList();

    // go through all instances, hashing them into lists corresponding to each class
    Enumeration enum = instances.enumerateInstances();
    while (enum.hasMoreElements()) {
      Instance instance = (Instance) enum.nextElement();
      if (instance.classIsMissing()) {
	System.err.println("Instance has missing class!!!");
	continue;
      }
      Double classValue = new Double(instance.classValue());

      // if this class has been seen, add instance to its list
      if (classInstanceMap.containsKey(classValue)) {
	ArrayList classInstanceList = (ArrayList) classInstanceMap.get(classValue);
	classInstanceList.add(instance);
      } else {  // create a new list of instances for a previously unseen class
	ArrayList classInstanceList = new ArrayList();
	classInstanceList.add(instance);
	classInstanceMap.put(classValue, classInstanceList);
	classValueList.add(classValue);
      }
    }

    // Create the desired number of random positive instances
    int numClasses = classInstanceMap.size();
    Random random = new Random();
    for (int i = 0; i < numPosPairs; i++) {
      // select a random class... TODO: probability must be proportional to the number of instances
      int class1 = random.nextInt(numClasses);
      ArrayList list = (ArrayList) classInstanceMap.get(classValueList.get(class1));
      int idx1 = random.nextInt(list.size());
      int idx2;
      do { 
	idx2 = random.nextInt(list.size());
      } while (idx1 == idx2);
      Instance instance1 = (Instance) list.get(idx1);
      Instance instance2 = (Instance) list.get(idx2);
      
      TrainingPair posPair = new TrainingPair(instance1, instance2, true, 0);
      pairList.add(posPair);
    }

    // Create negative diff-instances
    if (numClasses > 1) {
      random = new Random();
      for (int i = 0; i < numNegPairs; i++) {
	// select two random distinct classes
	int class1 = random.nextInt(numClasses);
	int class2 = random.nextInt(numClasses);
	while (class2 == class1) {
	  class2 = random.nextInt(numClasses);
	}
	ArrayList list1 = (ArrayList) classInstanceMap.get(classValueList.get(class1));
	Instance instance1 = (Instance) list1.get(random.nextInt(list1.size()));
	ArrayList list2 = (ArrayList) classInstanceMap.get(classValueList.get(class2));
	Instance instance2 = (Instance) list2.get(random.nextInt(list2.size()));
	TrainingPair negPair = new TrainingPair(instance1, instance2, false, 0);
	pairList.add(negPair);
      }
    }
    return pairList;
  }

  /** Print the heaviest-weighted attributes for a given set of weights
   */
  public void printTopAttributes(double[] weights, int n, int iteration) {
    // Print top weights - to be moved out into a separate function
    System.out.println(iteration + " top components:");
    int[] sortedIndeces = Utils.sort(weights);
    for (int i = sortedIndeces.length-1; i >  sortedIndeces.length-n && i >=0; i--) {
      int idx = sortedIndeces[i];
      System.out.println((sortedIndeces.length-1-i) + ": " +
			 idx + ":" + m_instances.attribute(idx).name()  + "(" + weights[idx] + ")");
    }
  }

}
