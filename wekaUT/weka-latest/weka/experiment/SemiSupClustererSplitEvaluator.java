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
 *    SemiSupClustererSplitEvaluator.java
 *    Copyright (C) 2002 Sugato Basu
 *
 */


package weka.experiment;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.clusterers.*;

/**
 * A SplitEvaluator that produces results for a semi-supervised clustering scheme
 * on a nominal class attribute.
 *
 * -W clustername <br>
 * Specify the full class name of the clusterer to evaluate. <p>
 *
 * -C class index <br>
 * The index of the class for which statistics are to
 * be output. (default 1) <p>
 *
 * @author Sugato Basu
 */

public class SemiSupClustererSplitEvaluator implements SplitEvaluator, 
  OptionHandler {
  
  /** The semi-supervised clusterer used for evaluation */
  protected Clusterer m_Clusterer = new MPCKMeans();

  /** Holds the statistics for the most recent application of the clusterer */
  protected String m_result = null;

  /** The clusterer options (if any) */
  protected String m_ClustererOptions = "";

  /** The clusterer version */
  protected String m_ClustererVersion = "";

  /** The length of a key */
  private static final int KEY_SIZE = 3;

  /** The length of a result */
  private static final int RESULT_SIZE = 13;

  /** Class index for information retrieval statistics (default 0) */
  private int m_IRclass = 0;

  /**
   * No args constructor.
   */
  public SemiSupClustererSplitEvaluator() {

    updateOptions();
  }

  /** Does nothing, since cluster evaluation does not allow additional measures */
  public void setAdditionalMeasures(String [] additionalMeasures){}
  
  /**
   * Returns a string describing this split evaluator
   * @return a description of the split evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return " A SplitEvaluator that produces results for a semi-supervised "
      + "clustering scheme on a nominal class attribute.";
  }

  /**
   * Returns an enumeration describing the available options..
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
	     "\tThe full class name of the clusterer.\n"
	      +"\teg: weka.clusterers.SimpleKMeans", 
	     "W", 1, 
	     "-W <class name>"));
    newVector.addElement(new Option(
	     "\tThe index of the class for which IR statistics\n" +
	     "\tare to be output. (default 1)",
	     "C", 1, 
	     "-C <index>"));

    if ((m_Clusterer != null) &&
	(m_Clusterer instanceof OptionHandler)) {
      newVector.addElement(new Option(
	     "",
	     "", 0, "\nOptions specific to clusterer "
	     + m_Clusterer.getClass().getName() + ":"));
      Enumeration enum = ((OptionHandler)m_Clusterer).listOptions();
      while (enum.hasMoreElements()) {
	newVector.addElement(enum.nextElement());
      }
    }
    return newVector.elements();
  }

  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -W classname <br>
   * Specify the full class name of the clusterer to evaluate. <p>
   *
   * -C class index <br>
   * The index of the class for which IR statistics are to
   * be output. (default 1) <p>
   *
   * All option after -- will be passed to the clusterer.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    String cName = Utils.getOption('W', options);
    if (cName.length() == 0) {
      throw new Exception("A clusterer must be specified with"
			  + " the -W option.");
    }
    // Do it first without options, so if an exception is thrown during
    // the option setting, listOptions will contain options for the actual
    // Clusterer.
    setClusterer(Clusterer.forName(cName, null));
    if (getClusterer() instanceof OptionHandler) {
      ((OptionHandler) getClusterer())
	.setOptions(Utils.partitionOptions(options));
      updateOptions();
    }

    String indexName = Utils.getOption('C', options);
    if (indexName.length() != 0) {
      m_IRclass = (new Integer(indexName)).intValue() - 1;
    } else {
      m_IRclass = 0;
    }
  }

  /**
   * Gets the current settings of the Clusterer.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] clustererOptions = new String [0];
    if ((m_Clusterer != null) && 
	(m_Clusterer instanceof OptionHandler)) {
      clustererOptions = ((OptionHandler)m_Clusterer).getOptions();
    }
    
    String [] options = new String [clustererOptions.length + 5];
    int current = 0;

    if (getClusterer() != null) {
      options[current++] = "-W";
      options[current++] = getClusterer().getClass().getName();
    }
    options[current++] = "-C"; 
    options[current++] = "" + (m_IRclass + 1);
    options[current++] = "--";

    System.arraycopy(clustererOptions, 0, options, current, 
		     clustererOptions.length);
    current += clustererOptions.length;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


  /**
   * Gets the data types of each of the key columns produced for a single run.
   * The number of key fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array containing objects of the type of each key column. The 
   * objects should be Strings, or Doubles.
   */
  public Object [] getKeyTypes() {

    Object [] keyTypes = new Object[KEY_SIZE];
    keyTypes[0] = "";
    keyTypes[1] = "";
    keyTypes[2] = "";
    return keyTypes;
  }

  /**
   * Gets the names of each of the key columns produced for a single run.
   * The number of key fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array containing the name of each key column
   */
  public String [] getKeyNames() {

    String [] keyNames = new String[KEY_SIZE];
    keyNames[0] = "Scheme";
    keyNames[1] = "Scheme_options";
    keyNames[2] = "Scheme_version_ID";
    return keyNames;
  }

  /**
   * Gets the key describing the current SplitEvaluator. For example
   * This may contain the name of the clusterer used for clusterer
   * predictive evaluation. The number of key fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array of objects containing the key.
   */
  public Object [] getKey(){

    Object [] key = new Object[KEY_SIZE];
    key[0] = m_Clusterer.getClass().getName();
    key[1] = m_ClustererOptions;
    key[2] = m_ClustererVersion;
    return key;
  }

  /**
   * Gets the data types of each of the result columns produced for a 
   * single run. The number of result fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array containing objects of the type of each result column. 
   * The objects should be Strings, or Doubles.
   */
  public Object [] getResultTypes() {
    int overall_length = RESULT_SIZE;
    Object [] resultTypes = new Object[overall_length];
    Double doub = new Double(0);
    int current = 0;
    
    // Unsupervised stats: 3
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Supervised stats: 2
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Training data stats: 2  
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // IR stats: 3
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Timing stats: 2
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Clusterer defined extras: 1
    resultTypes[current++] = "";

    if (current != overall_length) {
      throw new Error("ResultTypes didn't fit RESULT_SIZE");
    }
    return resultTypes;
  }

  /**
   * Gets the names of each of the result columns produced for a single run.
   * The number of result fields must be constant for a given SplitEvaluator.
   *
   * @return an array containing the name of each result column
   */
  public String [] getResultNames() {
    int overall_length = RESULT_SIZE;
    String [] resultNames = new String[overall_length];
    int current = 0;

    // Unsupervised stats: 3
    resultNames[current++] = "Purity";
    resultNames[current++] = "Entropy";
    resultNames[current++] = "Objective_function";

    // Supervised stats: 2
    resultNames[current++] = "KL_divergence";
    resultNames[current++] = "Mutual_information";

    // Training data stats: 2  
    resultNames[current++] = "SameClassPairs";
    resultNames[current++] = "DiffClassPairs";

    // IR stats: 3
    resultNames[current++] = "Pairwise_ir_precision";
    resultNames[current++] = "Pairwise_ir_recall";
    resultNames[current++] = "Pairwise_f_measure";

    // Timing stats: 2
    resultNames[current++] = "Time_training";
    resultNames[current++] = "Time_testing";

    // Clusterer defined extras: 1
    resultNames[current++] = "Summary";

    if (current != overall_length) {
      throw new Error("ResultNames didn't fit RESULT_SIZE");
    }
    return resultNames;
  }

  /** Dummy function, exists just for compatibility with SplitEvaluator interface
   */
  public Object [] getResult(Instances unlabeledTrain, Instances test) {
    try {
      return getResult(null, unlabeledTrain, test, test.numClasses(), -1); // labeled set is null
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param labeledTrainPairs the constraint pairs having labels on them
   * @param labeledTrain the labeled training Instances.
   * @param unlabeledData the unlabeled training (+ test for transductive) Instances.
   * @param test the testing Instances.
   * @param startingIndexOfTest from where test data starts in unlabeledData, useful if clustering is transductive
   * @return the results stored in an array. The objects stored in
   * the array may be Strings, Doubles, or null (for the missing value).
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(ArrayList labeledTrainPairs, Instances labeledTrain, Instances unlabeledData, Instances test, Instances unlabeledTest) throws Exception{
    if (m_Clusterer == null) {
      throw new WekaException("No clusterer has been specified");
    }
    if (!(m_Clusterer instanceof SemiSupClusterer)) {
      throw new WekaException("Clusterer should implement SemiSupClusterer interface!!\n"); // KLUGE (we could not make m_Clusterer of type SemiSupClusterer, since SemiSupClusterer is an interface and not an abstract class ... so we have to make the check here)
    }

    int overall_length = RESULT_SIZE;
    Object [] result = new Object[overall_length];
    long trainTimeStart = System.currentTimeMillis();

    if (m_Clusterer instanceof PCKMeans) {
      ((PCKMeans)m_Clusterer).buildClusterer(labeledTrainPairs, unlabeledData, labeledTrain, labeledTrain.numInstances()); // KLUGE: have to generalize later
    } else if (m_Clusterer instanceof MPCKMeans) {
      ((MPCKMeans)m_Clusterer).buildClusterer(labeledTrainPairs, unlabeledData, labeledTrain, labeledTrain.numClasses(), labeledTrain.numInstances()); // KLUGE: have to generalize later
    } else {
      throw new Exception ("Inappropriate clusterer: " + m_Clusterer.getClass().getName());
    } 

    //    ((SeededKMeans)m_Clusterer).printClusters();

    int numClusters = labeledTrain.numClasses();
    if (m_Clusterer instanceof SemiSupClusterer) {
      numClusters = ((SemiSupClusterer)m_Clusterer).getNumClusters();
    }

    SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(labeledTrainPairs, test, labeledTrain.numClasses(), numClusters);

    long trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
    long testTimeStart = System.currentTimeMillis();
    eval.evaluateModel(m_Clusterer, test, unlabeledTest);
    long testTimeElapsed = System.currentTimeMillis() - testTimeStart;
    m_result = eval.toSummaryString();

    // The results stored are all per instance -- can be multiplied by the
    // number of instances to get absolute numbers
    int current = 0;    

    // Unsupervised stats: 3
    result[current++] = new Double(eval.purity());
    result[current++] = new Double(eval.entropy());
    result[current++] = new Double(eval.objectiveFunction());

    // Supervised stats: 2
    result[current++] = new Double(eval.klDivergence());
    result[current++] = new Double(eval.mutualInformation());

    // Training data stats: 2
    result[current++] = new Double(eval.numSameClassPairs());
    result[current++] = new Double(eval.numDiffClassPairs());

    // IR stats: 3
    result[current++] = new Double(eval.pairwisePrecision());
    result[current++] = new Double(eval.pairwiseRecall());
    result[current++] = new Double(eval.pairwiseFMeasure());

    // Timing stats: 2
    result[current++] = new Double(trainTimeElapsed / 1000.0);
    result[current++] = new Double(testTimeElapsed / 1000.0);

    // Clusterer defined extras: 1
    if (m_Clusterer instanceof Summarizable) {
      result[current++] = ((Summarizable)m_Clusterer).toSummaryString();
    } else {
      result[current++] = null;
    }

    if (current != overall_length) {
      throw new Error("Results didn't fit RESULT_SIZE");
    }
    return result;
  }

  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param labeledTrain the labeled training Instances.
   * @param unlabeledTrain the unlabeled training Instances.
   * @param test the testing Instances.
   * @return the results stored in an array. The objects stored in
   * the array may be Strings, Doubles, or null (for the missing value).
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances labeledTrain, Instances unlabeledTrain, Instances test, int numClasses) 
    throws Exception {
    try {
      return getResult(labeledTrain, unlabeledTrain, test, test.numClasses(), -1); // labeled set is null
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }


  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param labeledTrain the labeled training Instances.
   * @param unlabeledData the unlabeled training (+ test for transductive) Instances.
   * @param test the testing Instances.
   * @param startingIndexOfTest from where test data starts in unlabeledData, useful if clustering is transductive
   * @return the results stored in an array. The objects stored in
   * the array may be Strings, Doubles, or null (for the missing value).
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances labeledTrain, Instances unlabeledData, Instances totalTrainWithLabels, Instances test, int startingIndexOfTest) 
    throws Exception {
    
    if (labeledTrain.classAttribute().type() != Attribute.NOMINAL) {
      throw new WekaException("Class attribute is not nominal!");
    }
    if (m_Clusterer == null) {
      throw new WekaException("No clusterer has been specified");
    }
    if (!(m_Clusterer instanceof SemiSupClusterer)) {
      throw new WekaException("Clusterer should implement SemiSupClusterer interface!!\n"); // KLUGE (we could not make m_Clusterer of type SemiSupClusterer, since SemiSupClusterer is an interface and not an abstract class ... so we have to make the check here)
    }

    int overall_length = RESULT_SIZE;
    Object [] result = new Object[overall_length];
    long trainTimeStart = System.currentTimeMillis();
    int classIndex = labeledTrain.numAttributes()-1; // assuming that the last attribute is always the class

    ((SeededKMeans)m_Clusterer).buildClusterer(labeledTrain, unlabeledData, classIndex, totalTrainWithLabels, startingIndexOfTest);

    int numClusters = totalTrainWithLabels.numClasses();
    if (m_Clusterer instanceof SemiSupClusterer) {
      numClusters = ((SemiSupClusterer)m_Clusterer).getNumClusters();
    }

    SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(test, totalTrainWithLabels.numClasses(), numClusters);

    long trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
    long testTimeStart = System.currentTimeMillis();
    Instances unlabeledTest = new Instances (test);
    unlabeledTest.deleteClassAttribute();
    eval.evaluateModel(m_Clusterer, test, unlabeledTest);
    long testTimeElapsed = System.currentTimeMillis() - testTimeStart;
    m_result = eval.toSummaryString();

    // The results stored are all per instance -- can be multiplied by the
    // number of instances to get absolute numbers
    int current = 0;    

    // Unsupervised stats: 3
    result[current++] = new Double(eval.purity());
    result[current++] = new Double(eval.entropy());
    result[current++] = new Double(eval.objectiveFunction());

    // Supervised stats: 2
    result[current++] = new Double(eval.klDivergence());
    result[current++] = new Double(eval.mutualInformation());

    // Training data stats: 2 - there are no training pairs in this case
    result[current++] = new Double(0);
    result[current++] = new Double(0);

    // IR stats: 3
    result[current++] = new Double(eval.pairwisePrecision());
    result[current++] = new Double(eval.pairwiseRecall());
    result[current++] = new Double(eval.pairwiseFMeasure());

    // Timing stats: 2
    result[current++] = new Double(trainTimeElapsed / 1000.0);
    result[current++] = new Double(testTimeElapsed / 1000.0);

    // Clusterer defined extras: 1
    if (m_Clusterer instanceof Summarizable) {
      result[current++] = ((Summarizable)m_Clusterer).toSummaryString();
    } else {
      result[current++] = null;
    }

    if (current != overall_length) {
      throw new Error("Results didn't fit RESULT_SIZE");
    }
    return result;
  }


  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param labeledTrain the labeled training Instances.
   * @param unlabeledTrain the unlabeled training Instances.
   * @param test the testing Instances.
   * @param startingIndexOfTest from where test data starts in unlabeledData, useful if clustering is transductive
   * @return the results stored in an array. The objects stored in
   * the array may be Strings, Doubles, or null (for the missing value).
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances labeledTrain, Instances unlabeledTrain, Instances test, int numClasses, int startingIndexOfTest) 
    throws Exception {
    
    if (labeledTrain.classAttribute().type() != Attribute.NOMINAL) {
      throw new WekaException("Class attribute is not nominal!");
    }
    if (m_Clusterer == null) {
      throw new WekaException("No clusterer has been specified");
    }
    if (!(m_Clusterer instanceof SemiSupClusterer)) {
      throw new WekaException("Clusterer should implement SemiSupClusterer interface!!\n"); // KLUGE (we could not make m_Clusterer of type SemiSupClusterer, since SemiSupClusterer is an interface and not an abstract class ... so we have to make the check here)
    }

    int overall_length = RESULT_SIZE;
    Object [] result = new Object[overall_length];
    long trainTimeStart = System.currentTimeMillis();
    int classIndex = labeledTrain.numAttributes()-1; // assuming that the last attribute is always the class

    ((SemiSupClusterer)m_Clusterer).buildClusterer(labeledTrain, unlabeledTrain, classIndex, numClasses, startingIndexOfTest);

    int numClusters = numClasses;
    if (m_Clusterer instanceof SemiSupClusterer) {
      numClusters = ((SemiSupClusterer)m_Clusterer).getNumClusters();
    }

    SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(test, numClasses, numClusters);

    long trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
    long testTimeStart = System.currentTimeMillis();
    Instances unlabeledTest = new Instances (test);
    unlabeledTest.deleteClassAttribute();
    eval.evaluateModel(m_Clusterer, test, unlabeledTest);
    long testTimeElapsed = System.currentTimeMillis() - testTimeStart;
    m_result = eval.toSummaryString();

    // The results stored are all per instance -- can be multiplied by the
    // number of instances to get absolute numbers
    int current = 0;    

    // Unsupervised stats: 3
    result[current++] = new Double(eval.purity());
    result[current++] = new Double(eval.entropy());
    result[current++] = new Double(eval.objectiveFunction());

    // Supervised stats: 2
    result[current++] = new Double(eval.klDivergence());
    result[current++] = new Double(eval.mutualInformation());

    // Training data stats: 2 - there are no training pairs in this case
    result[current++] = new Double(0);
    result[current++] = new Double(0);

    // IR stats: 3
    result[current++] = new Double(eval.pairwisePrecision());
    result[current++] = new Double(eval.pairwiseRecall());
    result[current++] = new Double(eval.pairwiseFMeasure());

    // Timing stats: 2
    result[current++] = new Double(trainTimeElapsed / 1000.0);
    result[current++] = new Double(testTimeElapsed / 1000.0);

    // Clusterer defined extras: 1
    if (m_Clusterer instanceof Summarizable) {
      result[current++] = ((Summarizable)m_Clusterer).toSummaryString();
    } else {
      result[current++] = null;
    }

    if (current != overall_length) {
      throw new Error("Results didn't fit RESULT_SIZE");
    }
    return result;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String clustererTipText() {
    return "The clusterer to use.";
  }

  /**
   * Get the value of Clusterer.
   *
   * @return Value of Clusterer.
   */
  public Clusterer getClusterer() {
    
    return m_Clusterer;
  }
  
  /**
   * Sets the clusterer.
   *
   * @param newClusterer the new clusterer to use.
   */
  public void setClusterer(Clusterer newClusterer) {
    
    m_Clusterer = newClusterer;
    updateOptions();
    
    System.err.println("SemiSupClustererSplitEvaluator: In set clusterer");
  }
  
  /**
   * Get the value of ClassForIRStatistics.
   * @return Value of ClassForIRStatistics.
   */
  public int getClassForIRStatistics() {
    
    return m_IRclass;
  }
  
  /**
   * Set the value of ClassForIRStatistics.
   * @param v  Value to assign to ClassForIRStatistics.
   */
  public void setClassForIRStatistics(int v) {
    
    m_IRclass = v;
  }
  
  /**
   * Updates the options that the current clusterer is using.
   */
  protected void updateOptions() {
    
    if (m_Clusterer instanceof OptionHandler) {
      m_ClustererOptions = Utils.joinOptions(((OptionHandler)m_Clusterer)
					      .getOptions());
    } else {
      m_ClustererOptions = "";
    }
    if (m_Clusterer instanceof Serializable) {
      ObjectStreamClass obs = ObjectStreamClass.lookup(m_Clusterer
						       .getClass());
      m_ClustererVersion = "" + obs.getSerialVersionUID();
    } else {
      m_ClustererVersion = "";
    }
  }

  /**
   * Set the Clusterer to use, given it's class name. A new clusterer will be
   * instantiated.
   *
   * @param newClusterer the Clusterer class name.
   * @exception Exception if the class name is invalid.
   */
  public void setClustererName(String newClustererName) throws Exception {

    try {
      setClusterer((Clusterer)Class.forName(newClustererName)
		    .newInstance());
    } catch (Exception ex) {
      throw new Exception("Can't find Clusterer with class name: "
			  + newClustererName);
    }
  }

  /**
   * Gets the raw output from the clusterer
   * @return the raw output from the clusterer
   */
  public String getRawResultOutput() {
    StringBuffer result = new StringBuffer();

    if (m_Clusterer == null) {
      return "<null> clusterer";
    }
    result.append(toString());
    result.append("Clusterer model: \n"+m_Clusterer.toString()+'\n');

    // append the performance statistics
    if (m_result != null) {
      result.append(m_result);
    }
    return result.toString();
  }

  /**
   * Returns a text description of the split evaluator.
   *
   * @return a text description of the split evaluator.
   */
  public String toString() {

    String result = "SemiSupClustererSplitEvaluator: ";
    if (m_Clusterer == null) {
      return result + "<null> clusterer";
    }
    return result + m_Clusterer.getClass().getName() + " " 
      + m_ClustererOptions + "(version " + m_ClustererVersion + ")";
  }
} // SemiSupClustererSplitEvaluator
