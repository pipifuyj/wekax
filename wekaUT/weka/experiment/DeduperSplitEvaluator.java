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
 *    DeduperSplitEvaluator.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */


package weka.experiment;

import java.io.*;
import java.util.*;

import weka.core.*;
import weka.deduping.*;


/**
 * A SplitEvaluator that produces results for a deduper scheme
 * on a nominal class attribute.
 *
 * -W classname <br>
 * Specify the full class name of the deduper to evaluate. <p>
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.2 $
 */
public class DeduperSplitEvaluator implements SplitEvaluator, 
  OptionHandler {
  
  /** The deduper used for evaluation */
  protected Deduper m_deduper = new BasicDeduper();

  /** Holds the statistics for the most recent application of the deduper */
  protected String m_result = null;

  /** The deduper options (if any) */
  protected String m_deduperOptions = "";

  /** The deduper version */
  protected String m_deduperVersion = "";

  /** The length of a key */
  private static final int KEY_SIZE = 3;

  /** The length of a result */
  private static final int RESULT_SIZE = 16;

  /**
   * No args constructor.
   */
  public DeduperSplitEvaluator() {

    updateOptions();
  }

  /**
   * Returns a string describing this split evaluator
   * @return a description of the split evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return " A SplitEvaluator that produces results for a deduper "
      +"scheme on a nominal class attribute.";
  }

  /** Does nothing, since deduping evaluation does not allow additional measures */
  public void setAdditionalMeasures(String [] additionalMeasures){}
  
  /**
   * Returns an enumeration describing the available options..
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
	     "\tThe full class name of the deduper.\n"
	      +"\teg: weka.dedupers.BasicDeduper", 
	     "W", 1, 
	     "-W <class name>"));

    if ((m_deduper != null) &&
	(m_deduper instanceof OptionHandler)) {
      newVector.addElement(new Option(
	     "",
	     "", 0, "\nOptions specific to dedupers "
	     + m_deduper.getClass().getName() + ":"));
      Enumeration enum = ((OptionHandler)m_deduper).listOptions();
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
   * Specify the full class name of the deduper to evaluate. <p>
   *
   * All option after -- will be passed to the deduper.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String cName = Utils.getOption('W', options);
    if (cName.length() == 0) {
      throw new Exception("A deduper must be specified with"
			  + " the -W option.");
    }
    // Do it first without options, so if an exception is thrown during
    // the option setting, listOptions will contain options for the actual
    // Deduper.
    setDeduper(Deduper.forName(cName, null));
    if (getDeduper() instanceof OptionHandler) {
      ((OptionHandler) getDeduper())
	.setOptions(Utils.partitionOptions(options));
      updateOptions();
    }
  }

  /**
   * Gets the current settings of the Deduper.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] deduperOptions = new String [0];
    if ((m_deduper != null) && 
	(m_deduper instanceof OptionHandler)) {
      deduperOptions = ((OptionHandler)m_deduper).getOptions();
    }
    
    String [] options = new String [deduperOptions.length + 5];
    int current = 0;

    if (getDeduper() != null) {
      options[current++] = "-W";
      options[current++] = getDeduper().getClass().getName();
    }
    options[current++] = "--";

    System.arraycopy(deduperOptions, 0, options, current, 
		     deduperOptions.length);
    current += deduperOptions.length;
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
   * This may contain the name of the deduper used for deduper
   * predictive evaluation. The number of key fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array of objects containing the key.
   */
  public Object [] getKey(){

    Object [] key = new Object[KEY_SIZE];
    key[0] = m_deduper.getClass().getName();
    key[1] = m_deduperOptions;
    key[2] = m_deduperVersion;
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
    resultTypes[current++] = doub;

    // Accuracy stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Dupe density stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;    

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Timing stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    if (current != overall_length) {
      throw new Error("ResultTypes didn't fit RESULT_SIZE");
    }
    return resultTypes;
  }

  /**
   * Gets the names of each of the result columns produced for a single run.
   * The number of result fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array containing the name of each result column
   */
  public String [] getResultNames() {
    int overall_length = RESULT_SIZE;
    String [] resultNames = new String[overall_length];
    int current = 0;
    resultNames[current++] = "Number_of_instances";

    // Accuracy stats
    resultNames[current++] = "Recall";
    resultNames[current++] = "Precision";
    resultNames[current++] = "Fmeasure";

    // Dupe density stats
    resultNames[current++] = "TotalPairsTrain";
    resultNames[current++] = "PotentialDupePairsTrain";
    resultNames[current++] = "ActualDupePairsTrain";
    resultNames[current++] = "PotentialNonDupePairsTrain";
    resultNames[current++] = "ActualNonDupePairsTrain";
    resultNames[current++] = "DupeNonDupeRatioTrain";
    resultNames[current++] = "DupeOveralProportionTrain";
    
    resultNames[current++] = "TotalPairsTest";
    resultNames[current++] = "DupePairsTest";
    resultNames[current++] = "DupeNonDupeRatioTest";

    // Timing stats
    resultNames[current++] = "Time_training";
    resultNames[current++] = "Time_testing";

    if (current != overall_length) {
      throw new Error("ResultNames didn't fit RESULT_SIZE");
    }
    return resultNames;
  }

  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param train the training Instances.
   * @param test the testing Instances.
   * @return the raw results stored in an array. The objects stored in
   * the array are object arrays, containing actual P/R/FM values for each point
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances trainData, Instances testData) 
    throws Exception {
    
    if (trainData.classAttribute().type() != Attribute.NOMINAL) {
      throw new Exception("Class attribute is not nominal!");
    }
    if (m_deduper == null) {
      throw new Exception("No deduper has been specified");
    }

    DedupingEvaluation eval = new DedupingEvaluation();
    eval.trainDeduper(m_deduper, trainData, testData);
    ArrayList rawResultList = eval.evaluateModel(m_deduper, testData);
    
    return rawResultList.toArray();
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String deduperTipText() {
    return "The deduper to use.";
  }

  /**
   * Get the value of Deduper.
   *
   * @return Value of Deduper.
   */
  public Deduper getDeduper() {
    return m_deduper;
  }
  
  /**
   * Sets the deduper.
   *
   * @param newDeduper the new deduper to use.
   */
  public void setDeduper(Deduper newDeduper) {
    m_deduper = newDeduper;
    updateOptions();
  }
  
  /**
   * Updates the options that the current deduper is using.
   */
  protected void updateOptions() {
    if (m_deduper instanceof OptionHandler) {
      m_deduperOptions = Utils.joinOptions(((OptionHandler)m_deduper)
					      .getOptions());
    } else {
      m_deduperOptions = "";
    }
    if (m_deduper instanceof Serializable) {
      ObjectStreamClass obs = ObjectStreamClass.lookup(m_deduper
						       .getClass());
      m_deduperVersion = "" + obs.getSerialVersionUID();
    } else {
      m_deduperVersion = "";
    }
  }

  /**
   * Set the Deduper to use, given it's class name. A new deduper will be
   * instantiated.
   *
   * @param newDeduper the Deduper class name.
   * @exception Exception if the class name is invalid.
   */
  public void setDeduperName(String newDeduperName) throws Exception {
    try {
      setDeduper((Deduper)Class.forName(newDeduperName)
		    .newInstance());
    } catch (Exception ex) {
      throw new Exception("Can't find Deduper with class name: "
			  + newDeduperName);
    }
  }

  /**
   * Gets the raw output from the deduper
   * @return the raw output from the deduper
   */
  public String getRawResultOutput() {
    StringBuffer result = new StringBuffer();

    if (m_deduper == null) {
      return "<null> deduper";
    }
    result.append(toString());
    result.append("Deduper model: \n"+m_deduper.toString()+'\n');

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

    String result = "DeduperSplitEvaluator: ";
    if (m_deduper == null) {
      return result + "<null> deduper";
    }
    return result + m_deduper.getClass().getName() + " " 
      + m_deduperOptions + "(version " + m_deduperVersion + ")";
  }
} // DeduperSplitEvaluator
