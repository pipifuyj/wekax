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

import weka.classifiers.*;
import weka.classifiers.sparse.*;
import weka.classifiers.functions.*;
import weka.core.*;

/** 
 * ClassifierMetricLearner - learns metric parameters by constructing
 * "difference instances" and then learning weights that classify same-class
 * instances as positive, and different-class instances as negative.
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.4 $
 */

public class ClassifierMetricLearner extends MetricLearner implements Serializable, OptionHandler {

  /** Classifier that is used for learning metric weights */
  protected Classifier m_classifier = new SVMlight();

  /** Class attribute for diff-instances can be either nominal or numeric */
  protected boolean m_isDiffClassNominal = true;

  /** The metric that the classifier was used to learn, useful for external-calculation based metrics */
  protected LearnableMetric m_metric = null;

  /** The pairwise selector used by the metric */
  protected PairwiseSelector m_selector = new HardPairwiseSelector();
  protected int m_numPosPairs = 200;
  protected int m_numNegPairs = 200;


  /** Create a new classifier metric learner 
   */
  public ClassifierMetricLearner() {
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
      m_metric = metric;
    }
    ArrayList pairList = m_selector.createPairList(instances, m_numPosPairs, m_numNegPairs, metric);
    Instances diffInstances = createDiffInstances(pairList, metric);

    if (diffInstances == null) {
      metric.m_trained = false;
      System.out.println("null diffInstances");
      return;
    }

    // BEGIN SANITY CHECK
    if (true) { 
      // dump diff-instances into a temporary file
      try {
	File diffDir = new File("/tmp/diff");
	diffDir.mkdir();
	String diffName = instances.relationName() + "." +
	  Utils.removeSubstring(metric.getClass().getName(), "weka.core.metrics.") + "." +
	  Utils.removeSubstring(m_selector.getClass().getName(), "weka.core.metrics.");

	if (m_selector instanceof HardPairwiseSelector) {
	  diffName = diffName + ((HardPairwiseSelector)m_selector).getNegativesMode().getSelectedTag().getReadable();
	  diffName = diffName + ((HardPairwiseSelector)m_selector).getPositivesMode().getSelectedTag().getReadable();
	} 
	diffInstances.setRelationName(diffName);
	PrintWriter writer = new PrintWriter(new BufferedOutputStream (new FileOutputStream(diffDir.getPath() + "/" +
											    diffName + ".arff")));
	writer.println(diffInstances.toString());
	writer.close();

	// Do a sanity check - dump out the diffInstances, and
	// evaluation classification with an SVM. 
	long trainTimeStart = System.currentTimeMillis();
	SVMlight classifier = new SVMlight();
	Evaluation eval = new Evaluation(diffInstances);
	eval.crossValidateModel(classifier, diffInstances, 3);
	writer = new PrintWriter(new BufferedOutputStream (new FileOutputStream(diffDir.getPath() + "/" +
										diffName + ".dat", true)));
	writer.println(eval.pctCorrect());
	writer.close();
	System.out.println("** Record Sanity:" + (System.currentTimeMillis() - trainTimeStart) + " ms; " +
			   eval.pctCorrect() + "% correct\t" +
			   eval.numFalseNegatives(0) + "(" + (eval.falseNegativeRate(0)*100) + "%) false negatives\t" +
			   eval.numFalsePositives(0) + "(" + (eval.falsePositiveRate(0)*100) + "%) false positives\t");
      
      } catch (Exception e) {
	e.printStackTrace();
	System.out.println(e.toString()); 
      }
    }
    // END SANITY CHECK


    System.out.println(getTimestamp()+ "  Building classifier: " +
		       Utils.removeSubstring(m_classifier.getClass().getName(), "weka.classifiers.") + " "
		       + concatStringArray(((OptionHandler)m_classifier).getOptions()));
    m_classifier.buildClassifier(diffInstances);

    // if we are learning coefficients, put them back into the distance metric
    if (!metric.getExternal()) {
      String classifierName = Utils.removeSubstring(m_classifier.getClass().getName(), "weka.classifiers");
      if (classifierName.equals("functions.LinearRegression")) {
	double[] coefficients = ((LinearRegression)m_classifier).coefficients();
	System.out.println("Learned coefficients " + coefficients.length);
	metric.setWeights(coefficients);
      }  else if (classifierName.equals("functions.SMO")) {
	FastVector weights = ((SMO)m_classifier).weights();
	double[] m_sparseWeights = (double[]) weights.elementAt(0);
	int[] m_sparseIndices = (int[]) weights.elementAt(1);
	double[] coefficients = new double[metric.getNumAttributes()];
	for (int i = 0; i < m_sparseIndices.length; i++) {
	  coefficients[m_sparseIndices[i]] = m_sparseWeights[i];
	} 
	metric.setWeights(coefficients);
      }
    } 
    System.out.println(getTimestamp() + "  Done building " +
		       Utils.removeSubstring(m_classifier.getClass().getName(), "weka.classifiers."));
    metric.m_trained = true;
  }

  /**
   * Set the classifier
   *
   * @param classifier the classifier
   */
  public void setClassifier (Classifier classifier) {
    m_classifier = classifier;
  }

  /**
   * Get the classifier
   *
   * @returns the classifierthat this metric employs
   */
  public Classifier getClassifier () {
    return m_classifier;
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


  /**
   * Use the Classifier for an estimation of similarity
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns sim an approximate similarity obtained from the classifier
   */
  public double getSimilarity(Instance instance1, Instance instance2) throws Exception{
    Instance diffInstance = m_metric.createDiffInstance(instance1, instance2);
    double d = (((DistributionClassifier)m_classifier).distributionForInstance(diffInstance))[1];
    return Math.exp(-d);
  }

  /**
   * Use the Classifier for an estimation of distance
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns  an approximate distance obtained from the classifier
   */
  public double getDistance(Instance instance1, Instance instance2) throws Exception{
    Instance diffInstance = m_metric.createDiffInstance(instance1, instance2);
    double d =  (((DistributionClassifier)m_classifier).distributionForInstance(diffInstance))[1];
    return d;
  }


  /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [40];
    
    int current = 0;

    options[current++] = "-p";
    options[current++] = "" + m_numPosPairs;
    options[current++] = "-n";
    options[current++] = "" + m_numNegPairs;

    options[current++] = "-S";
    options[current++] = Utils.removeSubstring(m_selector.getClass().getName(), "weka.core.metrics.");
    if (m_selector instanceof OptionHandler) {
      String[] selectorOptions = ((OptionHandler)m_selector).getOptions();
      for (int i = 0; i < selectorOptions.length; i++) {
	options[current++] = selectorOptions[i];
      }
    }
    
    options[current++] = "-B";
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


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -B classifierstring
   */
  public void setOptions(String[] options) throws Exception {
    String classifierString = Utils.getOption('B', options);
    if (classifierString.length() == 0) {
      throw new Exception("A classifier must be specified"
			  + " with the -B option.");
    }
    String [] classifierSpec = Utils.splitOptions(classifierString);
    if (classifierSpec.length == 0) {
      throw new Exception("Invalid classifier specification string");
    }
    String classifierName = classifierSpec[0];
    classifierSpec[0] = "";
    System.out.println("Classifier name: " + classifierName + "\nClassifier parameters: " + weka.classifiers.sparse.IBkMetric.concatStringArray(classifierSpec));
    setClassifier(Classifier.forName(classifierName, classifierSpec));
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
     Vector newVector = new Vector(1);

    newVector.addElement(new Option(
	      "\tFull class name of classifier to use, followed\n"
	      + "\tby scheme options. (required)\n"
	      + "\teg: \"weka.classifiers.bayes.NaiveBayes -D\"",
	      "B", 1, "-B <classifier specification>"));
    
    return newVector.elements();
  }

  /** Obtain a textual description of the metriclearner
   * @return a textual description of the metric learner
   */
  public String toString() {
    return new String("ClassifierMetricLearner " + concatStringArray(getOptions()));
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





