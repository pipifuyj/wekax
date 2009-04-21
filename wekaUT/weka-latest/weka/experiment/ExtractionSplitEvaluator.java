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
 *    ExtractionSplitEvaluator.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */


package weka.experiment;

import java.io.*;
import java.util.*;

import weka.core.*;
import weka.extraction.*;


/**
 * A SplitEvaluator that produces results for an extraction scheme
 *
 * -W classname <br>
 * Specify the full class name of the extractor to evaluate. <p>
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */
public class ExtractionSplitEvaluator implements SplitEvaluator, 
  OptionHandler {
  
  /** The extractor used for evaluation */
  protected Extractor m_extractor = null;  // TODO:  plug in "new SomeExtractor()" when ready

  /** Holds the statistics for the most recent application of the extractor */
  protected String m_result = null;

  /** The extractor options (if any) */
  protected String m_extractorOptions = "";

  /** The extractor version */
  protected String m_extractorVersion = "";

  /** The length of a key */
  private static final int KEY_SIZE = 3;

  /** The length of a result */
  private static final int RESULT_SIZE = 6;

  /**
   * No args constructor.
   */
  public ExtractionSplitEvaluator() {

    updateOptions();
  }

  /**
   * Returns a string describing this split evaluator
   * @return a description of the split evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return " A SplitEvaluator that produces results for a extractor "
      +"scheme on a nominal class attribute.";
  }

  /** Does nothing, since extraction evaluation does not allow additional measures */
  public void setAdditionalMeasures(String [] additionalMeasures){}
  
  /**
   * Returns an enumeration describing the available options..
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
	     "\tThe full class name of the extractor.\n"
	     +"\teg: weka.extractors.SomeExtractor",     // TODO:  plug in a name of some default extractor
	     "W", 1, 
	     "-W <class name>"));

    if ((m_extractor != null) &&
	(m_extractor instanceof OptionHandler)) {
      newVector.addElement(new Option(
	     "",
	     "", 0, "\nOptions specific to extractors "
	     + m_extractor.getClass().getName() + ":"));
      Enumeration enum = ((OptionHandler)m_extractor).listOptions();
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
   * Specify the full class name of the extractor to evaluate. <p>
   *
   * All option after -- will be passed to the extractor.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String cName = Utils.getOption('W', options);
    if (cName.length() == 0) {
      throw new Exception("A extractor must be specified with"
			  + " the -W option.");
    }
    // Do it first without options, so if an exception is thrown during
    // the option setting, listOptions will contain options for the actual
    // Extractor.
    setExtractor(Extractor.forName(cName, null));
    if (getExtractor() instanceof OptionHandler) {
      ((OptionHandler) getExtractor())
	.setOptions(Utils.partitionOptions(options));
      updateOptions();
    }
  }

  /**
   * Gets the current settings of the Extractor.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] extractorOptions = new String [0];
    if ((m_extractor != null) && 
	(m_extractor instanceof OptionHandler)) {
      extractorOptions = ((OptionHandler)m_extractor).getOptions();
    }
    
    String [] options = new String [extractorOptions.length + 5];
    int current = 0;

    if (getExtractor() != null) {
      options[current++] = "-W";
      options[current++] = getExtractor().getClass().getName();
    }
    options[current++] = "--";

    System.arraycopy(extractorOptions, 0, options, current, 
		     extractorOptions.length);
    current += extractorOptions.length;
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
   * This may contain the name of the extractor used for extractor
   * predictive evaluation. The number of key fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array of objects containing the key.
   */
  public Object [] getKey(){

    Object [] key = new Object[KEY_SIZE];
    key[0] = m_extractor.getClass().getName();
    key[1] = m_extractorOptions;
    key[2] = m_extractorVersion;
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
    resultTypes[current++] = doub;     // number of extracted fillers 

    resultTypes[current++] = doub;     // recall
    resultTypes[current++] = doub;     // precision
    resultTypes[current++] = doub;     // fmeasure

    // Timing stats
    resultTypes[current++] = doub;     // time training
    resultTypes[current++] = doub;     // time testing

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

    resultNames[current++] = "Recall";
    resultNames[current++] = "Precision";
    resultNames[current++] = "Fmeasure";

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
   * @param trainData the training Instances.
   * @param testData the testing Instances.
   * @return the raw results stored in an array. The objects stored in
   * the array are object arrays, containing actual P/R/FM values for each point
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances trainData, Instances testData) throws Exception {
    
    if (m_extractor == null) {
      throw new Exception("No extractor has been specified");
    }

    ExtractionEvaluation eval = new ExtractionEvaluation();
    eval.trainExtractor(m_extractor, trainData, testData);
    ArrayList rawResultList = eval.evaluateModel(m_extractor, testData);
    
    return rawResultList.toArray();
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String extractorTipText() {
    return "The extractor to use.";
  }

  /**
   * Get the value of Extractor.
   *
   * @return Value of Extractor.
   */
  public Extractor getExtractor() {
    return m_extractor;
  }
  
  /**
   * Sets the extractor.
   *
   * @param newExtractor the new extractor to use.
   */
  public void setExtractor(Extractor newExtractor) {
    m_extractor = newExtractor;
    updateOptions();
  }
  
  /**
   * Updates the options that the current extractor is using.
   */
  protected void updateOptions() {
    if (m_extractor instanceof OptionHandler) {
      m_extractorOptions = Utils.joinOptions(((OptionHandler)m_extractor)
					      .getOptions());
    } else {
      m_extractorOptions = "";
    }
    if (m_extractor instanceof Serializable) {
      ObjectStreamClass obs = ObjectStreamClass.lookup(m_extractor
						       .getClass());
      m_extractorVersion = "" + obs.getSerialVersionUID();
    } else {
      m_extractorVersion = "";
    }
  }

  /**
   * Set the Extractor to use, given it's class name. A new extractor will be
   * instantiated.
   *
   * @param newExtractor the Extractor class name.
   * @exception Exception if the class name is invalid.
   */
  public void setExtractorName(String newExtractorName) throws Exception {
    try {
      setExtractor((Extractor)Class.forName(newExtractorName)
		    .newInstance());
    } catch (Exception ex) {
      throw new Exception("Can't find Extractor with class name: "
			  + newExtractorName);
    }
  }

  /**
   * Gets the raw output from the extractor
   * @return the raw output from the extractor
   */
  public String getRawResultOutput() {
    StringBuffer result = new StringBuffer();

    if (m_extractor == null) {
      return "<null> extractor";
    }
    result.append(toString());
    result.append("Extractor model: \n"+m_extractor.toString()+'\n');

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

    String result = "ExtractionSplitEvaluator: ";
    if (m_extractor == null) {
      return result + "<null> extractor";
    }
    return result + m_extractor.getClass().getName() + " " 
      + m_extractorOptions + "(version " + m_extractorVersion + ")";
  }
} // ExtractionSplitEvaluator
