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
 *    SVMlight.java
 *    Copyright (C) 2002-2003 Mikhail Bilenko
 *
 *    TODO:
 *        - implement UpdateableClassifier
 *        - implement the remaining options for SVM-light
 *        - implement WeightedInstancesHandler
 *        - proper conversion from margin to distribution (see Zadrozny & Elkan, Wahba, Platt...)
 */

package weka.classifiers.sparse;

import weka.classifiers.Classifier;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import java.io.*;
import java.util.*;
import weka.core.*;


/**
 * <i> A wrapper for SVMlight package by Thorsten Joachims
 * For more information, see <p>
 *
 * http://www.cs.cornell.edu/People/tj/svm_light
 *
 * Valid options are:<p>
 *
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.9 $
 */
public class SVMlight extends DistributionClassifier
    implements OptionHandler {

  /** The training instances used for classification. */
  protected Instances m_train;
  
  /** Has the SVM been trained */ 
  protected boolean m_svmTrained = false;

  /** Output debugging information */
  protected boolean m_debug = false;

  /** Path to the directory where SVM-light executables are located */
  protected String m_binPath = new String("/u/ml/software/svm_light/");

  /** Path to the directory where temporary files will be stored */
  protected String m_tempDirPath = new String("/var/local/tmp/");
  protected File m_tempDirFile = null;

  /** Name of the temporary file where training data will be dumped temporarily */
  protected String m_trainFilenameBase = new String("SVMtrain");
  protected String m_trainFilename = null;

  /** Name of the temporary file where a test instance is dumped if buffered IO is not used */
  protected String m_testFilenameBase = new String("SVMtest");
  protected String m_testFilename = null;

  /** Name of the file where a model will be temporarily created*/
  protected String m_modelFilenameBase = new String("SVMmodel");
  protected String m_modelFilename = null;
  
  /** Name of the file where predictions will be temporarily stored unless buffered IO is used*/
  protected String m_predictionFilenameBase = new String("SVMpredict");
  protected String m_predictionFilename = null;

  /** SVM-light predictions are positive or negative margins; to convert
   * to a distribution we need min/max margin values... */ 
  protected double m_maxMargin = -45;
  protected double m_minMargin = 45;
  protected boolean m_autoBounds = false;

  /** Is classification done via temporary files or via a buffer? */
  protected boolean m_bufferedMode = true;
  protected BufferedReader m_procReader = null;
  protected BufferedWriter m_procWriter = null;


  /**********************/
  /** SVM-light options */

  /** verbosity level */
  protected int m_verbosityLevel = 1;

  /** SVM-light can work in classification, regression and preference ranking modes */
  public static final int SVM_MODE_CLASSIFICATION = 1;
  public static final int SVM_MODE_REGRESSION = 2;
  public static final int SVM_MODE_PREFERENCE_RANKING = 4;
  public static final Tag[] TAGS_SVM_MODE = {
    new Tag(SVM_MODE_CLASSIFICATION, "Classification"),
    new Tag(SVM_MODE_REGRESSION, "Regression"),
    new Tag(SVM_MODE_PREFERENCE_RANKING, "Preference ranking")
  };
  protected int m_mode = SVM_MODE_CLASSIFICATION;

  /** trade-off between training error and margin (default 0 corresponds to [avg. x*x]^-1) */
  protected double m_C = 0;      
  
  /** Epsilon width of tube for regression */
  protected double m_width = 0.1;

  /** Cost: cost-factor, by which training errors on positive examples outweight errors on negative examples */
  protected double m_costFactor = 1;

  /** Use biased hyperplane (i.e. x*w+b>0) instead of unbiased hyperplane (i.e. x*w>0) */
  protected boolean m_biased = true;

  /** remove inconsistent training examples and retrain */
  protected boolean m_removeInconsistentExamples = false;

  /** Kernel type */ 
  public static final int KERNEL_LINEAR = 1;
  public static final int KERNEL_POLYNOMIAL = 2;
  public static final int KERNEL_RBF = 4;
  public static final int KERNEL_SIGMOID_TANH = 8;
  public static final Tag[] TAGS_KERNEL_TYPE = {
    new Tag(KERNEL_LINEAR, "Linear"),
    new Tag(KERNEL_POLYNOMIAL, "Polynomial (s a*b+c)^d"),
    new Tag(KERNEL_RBF, "Radial basis function exp(-gamma ||a-b||^2)"),
    new Tag(KERNEL_SIGMOID_TANH, "Sigmoid tanh(s a*b + c)")
  };
  protected int m_kernelType = KERNEL_RBF;

  /** Parameter d in polynomial kernel */
  protected int m_d = 3;
  /** Parameter gamma in rbf kernel */
  protected double m_gamma = 1;
  /** Parameter s in sigmoid/polynomial kernel */
  protected double m_s = 1;
  /** parameter c in sigmoid/poly kernel */
  protected double m_c1 = 1; 


  /** A default constructor */
  public SVMlight() {
  }
  
  /** Take care of closing the SVM-light process before the object is destroyed
   */
  protected void finalize() {
    cleanupIO();
  }

  /** The buffered version of SVM-light needs to release some I/O resources
   * before exiting
   */
  protected void cleanupIO() {
    try {
      // kill the svm_classify_std process
      if (m_procWriter != null) { 
	m_procWriter.close();
      }
      if (m_procReader != null) { 
	m_procReader.close();
      }
      m_procReader = null;
      m_procWriter = null;

      // delete the model file
      if (!m_debug && (m_modelFilename != null)) {
	File modelFile = new File(m_modelFilename);
	modelFile.delete();
      }
	
    } catch (Exception e) {
      System.out.println("Problems when cleaning up IO");
      e.printStackTrace();
    }
  }
			    
  
  /**
   * Generates the classifier.
   *
   * @param instances set of instances serving as training data 
   * @exception Exception if the classifier has not been generated successfully
   */

  public void buildClassifier(Instances instances) throws Exception {
    if (instances.classIndex() < 0) {
      throw new Exception ("No class attribute assigned to instances.");
    }
    if (instances.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle string attributes.");
    }

    int numClasses = instances.numClasses();
    if (numClasses != 2) {
      throw new Exception("Training data should have two classes; has " + numClasses + " classes");
    }

    // if a classifier has been built, clean up the IO
    if (m_bufferedMode && m_procWriter != null) {
      cleanupIO();
    }

    // create a working copy of training data
    m_tempDirFile = new File(m_tempDirPath);
    m_train = new Instances(instances, 0, instances.numInstances());
    
    // Unlike most Weka classifiers, we are *not* throwing away training
    // instances with missing class, since they may be used for transduction.
    // If it is desired to avoid transduction and throw out unlabeled data,
    // uncomment the following line:
    // m_train.deleteWithMissingClass();
    
    // Convert training instances into SVMlight format and dump into a training file
    dumpTrainingData(m_train);

    // Train the model 
    trainSVMlight();

    // set min and max margin if desired
    if (m_autoBounds) {
      setBounds(instances);
    }
    
  }

  /** Set the bounds using "extreme" training examples - TODO!*/
  protected void setBounds(Instances data) {
    try {
      // get the minimum margin 
      double[] values = new double[data.numAttributes()];
      Instance zeroInstance = new Instance(1.0, values);
      zeroInstance.setDataset(data);
      
      if (!m_bufferedMode) {
	File testFile = File.createTempFile(m_testFilenameBase, ".dat", m_tempDirFile);
	if (!m_debug) { 
	  testFile.deleteOnExit();
	}
	dumpInstance(zeroInstance, testFile);
      }
      double minMargin = classifySVMlight(zeroInstance);
      setMinMargin(minMargin);


      // get the maximum margin
      double maxMargin = 0;
      for (int i = 0; i < data.numInstances(); i++) {
	Instance instance = data.instance(i);

	// we only care about positive examples
	if (instance.classValue() == 0) { 
	  if (!m_bufferedMode) {
	    File testFile = File.createTempFile(m_testFilenameBase, ".dat", m_tempDirFile);
	    if (!m_debug) { 
	      testFile.deleteOnExit();
	    }
	    dumpInstance(zeroInstance, testFile);
	  }
	  double margin = classifySVMlight(instance);
	  if (margin < maxMargin) {
	    maxMargin = margin;
	  }
	}
	setMaxMargin(maxMargin);
      }
      System.out.println("xxxxx  MINMARGIN=" + minMargin + "\tMAX_MARGIN=" + maxMargin);
    } catch (Exception e) {
      System.err.println("Problems obtaining automatic margins: " + e);
      e.printStackTrace();
    }
  } 


  /**
   * Dump training instances into a file in SVM-light format
   * @param instances the training instances
   * @param filename name of the file where instance will be dumped
   */
  protected void dumpTrainingData(Instances instances) {
    try {
      File trainFile = File.createTempFile(m_trainFilenameBase, ".dat", m_tempDirFile);
      if (!m_debug) { 
	trainFile.deleteOnExit();
      }
      m_trainFilename = trainFile.getPath();
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(trainFile)));
      int classIdx = instances.classIndex();
      
      // Go through all instances
      Enumeration enum = instances.enumerateInstances();
      while (enum.hasMoreElements()) {
	Instance trainInstance = (Instance) enum.nextElement();

	// output the class value
	double classValue = 0;
	if (!trainInstance.classIsMissing()) {
	  classValue = trainInstance.classValue();
	  if (classValue == 0) {
	    classValue = -1;
	  } else {
	    classValue = 1;
	  }
	}
	writer.print((int)classValue + " ");

	// output the attributes; iterating using numValues() skips 'missing' values for SparseInstances
	for (int j = 0; j < trainInstance.numValues(); j++) {
	  Attribute attribute = trainInstance.attributeSparse(j);
	  // Attribute index must be greater than 0
	  int attrIdx = attribute.index();
	  if (attrIdx != classIdx) {
	    writer.print((attrIdx+1) + ":" + trainInstance.value(attrIdx) + " ");
	  }
	}
	writer.println();
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error when dumping training instances: " + e);
      e.printStackTrace();
    }
  }

  /**
   * Dump a single instance into a file in SVM-light format
   * @param instance an instance
   * @param file the file where instance will be dumped
   */
  protected void dumpInstance(Instance instance, File file) {
    try { 
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(file)));
      
      // output a dummy class value
      int classIdx = instance.classIndex();
      writer.print(Integer.MAX_VALUE + " ");

      // output the attributes; iterating using numValues skips 'missing' values for SparseInstances
      for (int j = 0; j < instance.numValues(); j++) {
	Attribute attribute = instance.attributeSparse(j);
	int attrIdx = attribute.index();
	if (attrIdx != classIdx) {
	  writer.print((attrIdx+1) + ":" + instance.value(attrIdx) + " ");
	}
      }
      writer.println();
      writer.close();
    } catch (Exception e) {
      System.err.println("Error when dumping instance: " + e);
      e.printStackTrace();
    }
  }
  
  /** Launch an SVM-light process assuming that the training data has been dumped
   */
  protected void trainSVMlight() {
    try {
      String command = new String(m_binPath + "svm_learn");

      // append all options for SVM-light
      command = command + " -v " + m_verbosityLevel;

      switch (m_mode) {
      case SVM_MODE_CLASSIFICATION:
	command = command + " -z c";
	break;
      case SVM_MODE_REGRESSION:
	command = command + " -z r";
	command = command + " -w " + m_width;
	break;
      case SVM_MODE_PREFERENCE_RANKING:
	command = command + " -z p";
	break;
      default:
	throw new Exception("Unknown mode: " + m_mode);
      }

      command = command + " -c " + m_C;
      command = command + " -j " + m_costFactor;
      command = command + " -b " + (m_biased ? 1 : 0);
      command = command + " -i " + (m_removeInconsistentExamples ? 1 : 0);
      switch(m_kernelType) {
      case KERNEL_LINEAR:
	command = command + " -t 0";
	break;
      case KERNEL_POLYNOMIAL:
	command = command + " -t 1";
	command = command + " -d " + m_d;
	command = command + " -s " + m_s;
	command = command + " -r " + m_c1;
	break;
      case KERNEL_RBF:
	command = command + " -t 2";
	command = command + " -g " + m_gamma;
	break;
      case KERNEL_SIGMOID_TANH:
	command = command + " -t 3";
	command = command + " -s " + m_s;
	command = command + " -r " + m_c1;
	break;
      default:
	throw new Exception("Unknown kernel type: " + m_kernelType);
      }

      // create the model file
      File modelFile = File.createTempFile(m_modelFilenameBase, ".dat", m_tempDirFile);
      if (!m_debug) { 
	modelFile.deleteOnExit();
      }
      m_modelFilename = modelFile.getPath();
      
      command = command + " " + m_trainFilename + " " + m_modelFilename;
      if (m_debug) {
	System.out.println("Executing SVMlight: \n\t" + command);
      }
      Process proc = Runtime.getRuntime().exec(command);

      // read the training output
      if (proc != null){
	BufferedReader procOutput  = new BufferedReader(new InputStreamReader(proc.getInputStream()));
	try {
	  String line;
	  while ((line = procOutput.readLine()) != null){
	    if (m_debug) { 
	      System.out.println("SVM:  " + line);
	    }
	  }
	} catch (Exception e) {
	  System.err.println("Problems trapping output in debug mode:");
	  e.printStackTrace();
	  System.out.println(e);
	}
      }
      
      int exitValue = proc.waitFor();
      if (exitValue != 0) {
	throw new Exception("Problems training SVM-light:  process returned value " + exitValue);
      }
      // delete the training file
      File trainFile = new File(m_trainFilename);
      //      trainFile.delete();
      m_svmTrained = true;
    } catch (Exception e) {
      System.out.println("Problem training: ");
      e.printStackTrace();
      System.err.println(e);
    }
  } 


  /** Launch an SVM-light process and classify a given instance
   * @param instance an instance that must be classified
   */
  protected double classifySVMlight(Instance instance) {
    double prediction = Double.MIN_VALUE;
    String lineIn = null;
    StringBuffer instanceString = new StringBuffer();
    try {

      if (m_bufferedMode) {
	// if this is the first time classify() is called, initialize the classifier process
	if (m_procWriter == null) {  
	  String command = new String(m_binPath + "svm_classify_std -v " + m_verbosityLevel + " " + m_modelFilename);
	  if (m_debug) {
	    System.out.println("Executing \"" + command + "\"");
	  }
	  Process proc = Runtime.getRuntime().exec(command);
	  m_procReader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
	  m_procWriter = new BufferedWriter(new OutputStreamWriter(proc.getOutputStream()));
	  System.out.println(m_procReader.readLine());
	  System.out.println(m_procReader.readLine());
	}

	// pass the instance to SVMlight process

	// output a bogus class value
	instanceString.append(Integer.MAX_VALUE + " ");

	// output the attributes; iterating using numValues skips 'missing' values for SparseInstances
	int classIdx = instance.classIndex();
	for (int j = 0; j < instance.numValues(); j++) {
	  Attribute attribute = instance.attributeSparse(j);
	  int attrIdx = attribute.index();
	  if (attrIdx != classIdx) {
	    instanceString.append((attrIdx+1) + ":" + instance.value(attrIdx) + " ");
	  }
	}
	instanceString.append("\n");
	if (m_debug) {
	  System.out.println("Sending " + instance);
	  System.out.flush();
	}
	
	m_procWriter.write(instanceString.toString());
	m_procWriter.flush();

	lineIn = m_procReader.readLine();
	if (lineIn == null) {
	  throw new Exception("Got null prediction from SVMlight!");
	} 
	prediction = Double.parseDouble(lineIn);
	if (m_debug) {
	  System.out.println("Got " + prediction);
	}
	
      } else {  	// Non-buffered IO, a temporary test file is used for the test instance

	// create a temporary file where the test instance is dumped
	File testFile = File.createTempFile(m_testFilenameBase, ".dat", m_tempDirFile);
	if (!m_debug) { 
	  testFile.deleteOnExit();
	}
	m_testFilename = testFile.getPath();
	dumpInstance(instance, testFile);

	// create a temporary file where the SVMlight output (prediction) will be stored
	File predictionFile = File.createTempFile(m_predictionFilenameBase, ".dat", m_tempDirFile);
	if (!m_debug) { 
	  predictionFile.deleteOnExit();
	}
	m_predictionFilename = predictionFile.getPath();

	// run svm_classify
	String command = new String(m_binPath + "svm_classify -v " + m_verbosityLevel + " "
						 + m_testFilename + " "
						 + m_modelFilename + " "
						 + m_predictionFilename );
	Process proc = Runtime.getRuntime().exec(command);
	int exitValue = proc.waitFor();
	if (exitValue != 0) {
	  throw new Exception("Problems running SVM-light:  process returned value " + exitValue);
	}
	prediction = readPrediction(predictionFile);
	testFile.delete();
	predictionFile.delete();
      }
    } catch (Exception e) {
      System.out.println("Got from SVM-light: " + lineIn);
      System.err.println(e);
      e.printStackTrace();
    }
    return prediction;
  }

  /** Read the prediction of SVM-light
   * @param file file where the prediction is stored
   */
  protected double readPrediction(File file) {
    double result = Double.MIN_VALUE;
    try {
      BufferedReader r = new BufferedReader(new FileReader(file));
      String line = r.readLine();
      if (line == null) {
	throw new Exception("Empty prediction file " + file.getPath());
      }
      result =  Double.parseDouble(line);
    } catch (Exception e) {
      System.err.println("Error reading the prediction file: " + e);
    }
    return result;
  } 
  
  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if an error occurred during the prediction
   */
  public double [] distributionForInstance(Instance instance) throws Exception{
    if (!m_svmTrained) {
      throw new Exception("SVM has not been trained!");
    }

    // compute prediction
    double margin = classifySVMlight(instance);
    double[] predictions = new double[2];

    predictions[0] = 1 - (margin - m_maxMargin)/(m_minMargin - m_maxMargin);
    if (predictions[0] > 1) {
      //        System.out.println("overflow: " + predictions[0]);
      predictions[0] = 1;
    }
    if (predictions[0] < 0)  {
      //        System.out.println("underflow: " + predictions[0]);
      predictions[0] = 0;
    }

    predictions[1] = 1- predictions[0];

    if (m_debug) {
      System.out.println("\t\tMargin: " + margin + "\tDistribution: {" + predictions[0] + ",\t" + predictions[1] + "}");
    } 
    return predictions;
  }


  /** Check whether the SVM has been trained
   * @return true if the SVM has been train and is ready to classify instances
   */
  public boolean trained() {
    return m_svmTrained;
  } 
 
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
				    "\tOutput debug information",
				    "D", 0, "-D"));
    return newVector.elements();
  }

  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -D <br>
   * output debugging information <p>
   *
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    setDebug(Utils.getFlag('D', options));

    String verbosityString = Utils.getOption('v', options);
    if (verbosityString.length() != 0) {
      setVerbosityLevel(Integer.parseInt(verbosityString));
    }

    if (Utils.getFlag('A', options)) {
      setAutoBounds(true);
    } else {
      String minMarginString = Utils.getOption('n', options);
      if (minMarginString.length() != 0) {
	setMinMargin(Double.parseDouble(minMarginString));
      }
      
      String maxMarginString = Utils.getOption('m', options);
      if (maxMarginString.length() != 0) {
	setMaxMargin(Double.parseDouble(maxMarginString));
      }
    }

    if (Utils.getFlag('C', options)) {
      setMode(new SelectedTag(SVM_MODE_CLASSIFICATION, TAGS_SVM_MODE));
    } else if (Utils.getFlag('R', options)) {
      setMode(new SelectedTag(SVM_MODE_REGRESSION, TAGS_SVM_MODE));
      String widthString = Utils.getOption('w', options);
      if (widthString.length() != 0) {
	setWidth(Double.parseDouble(widthString));
      } 
    } else if (Utils.getFlag('P', options)) {
      setMode(new SelectedTag(SVM_MODE_PREFERENCE_RANKING, TAGS_SVM_MODE));
    }

    String cString = Utils.getOption('c', options);
    if (cString.length() != 0) {
      setC(Double.parseDouble(cString));
    }

    String costFactorString = Utils.getOption('j', options);
    if (costFactorString.length() != 0) {
      setCostFactor(Double.parseDouble(costFactorString));
    }

    setBiased(Utils.getFlag('b', options));
    setRemoveInconsistentExamples(Utils.getFlag('i', options));
    

    // kernel-type related options
    if (Utils.getFlag('L', options)) {
      setKernelType(new SelectedTag(KERNEL_LINEAR, TAGS_KERNEL_TYPE));
    } else if (Utils.getFlag('O', options)) {
      setKernelType(new SelectedTag(KERNEL_POLYNOMIAL, TAGS_KERNEL_TYPE));
      String dString = Utils.getOption('d', options);
      if (dString.length() != 0) {
	setD(Integer.parseInt(dString));
      }
      String sString = Utils.getOption('s', options);
      if (sString.length() != 0) {
	setS(Double.parseDouble(sString));
      }
      String c1String = Utils.getOption('r', options);
      if (c1String.length() != 0) {
	setC1(Double.parseDouble(c1String));
      }
    } else if (Utils.getFlag('B', options)) {
      setKernelType(new SelectedTag(KERNEL_RBF, TAGS_KERNEL_TYPE));
      String gammaString = Utils.getOption('g', options);
      if (gammaString.length() != 0) {
	setC1(Double.parseDouble(gammaString));
      }
    } else if (Utils.getFlag('S', options)) {
      setKernelType(new SelectedTag(KERNEL_SIGMOID_TANH, TAGS_KERNEL_TYPE));
      String sString = Utils.getOption('s', options);
      if (sString.length() != 0) {
	setS(Double.parseDouble(sString));
      }
      String c1String = Utils.getOption('r', options);
      if (c1String.length() != 0) {
	setC1(Double.parseDouble(c1String));
      }
    }

    String binPathString = Utils.getOption('p', options);
    if (binPathString.length() != 0) {
      setBinPath(binPathString);
    }

    Utils.checkForRemainingOptions(options);
  }


  /**
   * Gets the current settings 
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [20];
    int current = 0;

    if (m_debug) {
      options[current++] = "-D";
    }

    options[current++] = "-v";
    options[current++] = "" + m_verbosityLevel;

    if (m_autoBounds) {
      options[current++] = "-A";
    } else { 
      options[current++] = "-n";
      options[current++] = "" + m_minMargin;
      options[current++] = "-m";
      options[current++] = "" + m_maxMargin;
    }
    
    switch(m_mode) {
    case SVM_MODE_CLASSIFICATION:
      options[current++] = "-C";
      break;
    case SVM_MODE_REGRESSION:
      options[current++] = "-R";
      options[current++] = "-w";
      options[current++] = "" + m_width;
      break;
    case SVM_MODE_PREFERENCE_RANKING:
      options[current++] = "-P";
      break;
    default:
      System.err.println("UNKNOWN MODE: " + m_mode);
    }

    options[current++] = "-c";
    options[current++] = "" + m_C;

    options[current++] = "-j";
    options[current++] = "" + m_costFactor;
    
    if (m_biased) {
      options[current++] = "-b";
    }
    if (m_removeInconsistentExamples) {
      options[current++] = "-i";
    }

    switch (m_kernelType) {
    case KERNEL_LINEAR:
      options[current++] = "-L";
      break;
    case KERNEL_POLYNOMIAL:
      options[current++] = "-O";
      options[current++] = "-d";
      options[current++] = "" + m_d;
      options[current++] = "-s";
      options[current++] = "" + m_s;
      options[current++] = "-r";
      options[current++] = "" + m_c1;
      break;
    case KERNEL_RBF:
      options[current++] = "-B";
      options[current++] = "-g";
      options[current++] = "" + m_gamma;
      break;
    case KERNEL_SIGMOID_TANH:
      options[current++] = "-S";
      options[current++] = "-s";
      options[current++] = "" + m_s;
      options[current++] = "-r";
      options[current++] = "" + m_c1;
      break;
    default:
      System.err.println("UNKNOWN KERNEL TYPE: " + m_kernelType);
    }

    options[current++] = "-p";
    options[current++] = m_binPath;
    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /** Turn debugging output on/off
   * @param debug if true, SVM-light output and other debugging info will be printed
   */
  public void setDebug(boolean debug) {
    m_debug = debug;
  }

  /** See whether debugging output is on/off
   * @returns if true, SVM-light output and other debugging info will be printed
   */
  public boolean getDebug() {
    return m_debug;
  }

  /** Set SVM-light to operate via in/out bufffers or via temporary files
   * @param bufferedMode if true, SVM-light classification is performed via stdin/stdout
   */
  public void setBufferedMode(boolean bufferedMode) {
    m_bufferedMode = bufferedMode;
  }

  /** See whether SVM-light is operating via in/out bufffers or via temporary files
   * @returns if true, SVM-light classification is performed via stdin/stdout
   */
  public boolean getBufferedMode() {
    return m_bufferedMode;
  }

  /** Set verbosity level, can be anything between 0 and 3
   * @param verbosity Verbosity level for SVM-light
   */
  public void setVerbosityLevel(int verbosity) {
    m_verbosityLevel = verbosity;
  }

  /** Get verbosity level, can be anything between 0 and 3
   * @param verbosity Verbosity level for SVM-light
   */
  public int getVerbosityLevel() {
    return m_verbosityLevel;
  }

  /** Set the mode of the SVM
   * @param mode one of classification, regression and preference ranking
   */
  public void setMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_SVM_MODE) {
      m_mode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the SVM-light mode
   * @return one of  classification, regression and preference ranking
   */
  public SelectedTag getMode() {
    return new SelectedTag(m_mode, TAGS_SVM_MODE);
  }

  /** Set the epsilon width of tube for regression  */
  public void setWidth(double width) {
    m_width = width;
  }
  /** Get the epsilon width of tube for regression  */
  public double getWidth() {
    return m_width;
  }

  /** Set the trade-off between training error and margin (default 0 corresponds to [avg. x*x]^-1) */
  public void setC(double C) {
    m_C = C;
  }
  /** Get the trade-off between training error and margin (default 0 corresponds to [avg. x*x]^-1) */
  public double getC() {
    return m_C;
  }


  /** Set cost-factor, by which training errors on positive examples outweight errors on negative examples */
  public void setCostFactor(double costFactor) {
    m_costFactor = costFactor;
  }
  /** Get cost-factor, by which training errors on positive examples outweight errors on negative examples */
  public double getCostFactor() {
    return m_costFactor;
  }

  /** Set whether the hyperplane is biased (i.e. x*w+b>0) instead of unbiased hyperplane (i.e. x*w>0)
   * @param biased if true, the hyperplane will be biased
   */
  public void setBiased(boolean biased) {
    m_biased = biased;
  }

  /** Get whether the hyperplane is biased (i.e. x*w+b>0) instead of unbiased hyperplane (i.e. x*w>0)
   * @returns if true, the hyperplane will be biased
   */
  public boolean getBiased() {
    return m_biased;
  }

  /** Set whether the inconsistent examples are removed and retraining follows
   * @param removeInconsistentExamples
   */
  public void setRemoveInconsistentExamples(boolean removeInconsistentExamples) {
    m_removeInconsistentExamples = removeInconsistentExamples;
  } 

  /** Get whether the inconsistent examples are removed and retraining follows
   * @returns removeInconsistentExamples
   */
  public boolean getRemoveInconsistentExamples() {
    return m_removeInconsistentExamples;
  } 

  /** Set the kernel type for SVM-light
   * @param type one of the kernel types 
   */
  public void setKernelType(SelectedTag kernelType) {
    if (kernelType.getTags() == TAGS_KERNEL_TYPE) {
      m_kernelType = kernelType.getSelectedTag().getID();
    }
  }

  /** Get the SVM-light kernel type
   * @return kernel type 
   */
  public SelectedTag getKernelType() {
    return new SelectedTag(m_kernelType, TAGS_KERNEL_TYPE);
  }


  /** Set parameter d in polynomial kernel */
  public void setD(int d) {
    m_d = d;
  }
  /** Get parameter d in polynomial kernel */
  public int getD() {
    return m_d;
  }
  
  /** Set parameter gamma in rbf kernel */
  public void setGamma(double gamma) {
    m_gamma = gamma;
  }
  /** Get parameter gamma in rbf kernel */
  public double getGamma() {
    return m_gamma;
  }

  /** Set parameter s in sigmoid/polynomial kernel */
  public void setS(double s) {
    m_s = s;
  }
  /** Get parameter s in sigmoid/polynomial kernel */
  public double getS() {
    return m_s;
  }

  /** Set parameter c in sigmoid/poly kernel */
  public void setC1(double c1) {
    m_c1 = c1;
  } 
  /** Get parameter c in sigmoid/poly kernel */
  public double getC1() {
    return m_c1;
  } 


  /** Set the maxMargin that an SVM can return */
  public void setMaxMargin(double maxMargin) {
    m_maxMargin = maxMargin;
  }
  /** Get  the maxMargin that an SVM can return */
  public double getMaxMargin() {
    return m_maxMargin;
  }

  /** Set the minMargin that an SVM can return */
  public void setMinMargin(double minMargin) {
    m_minMargin = minMargin;
  }
  /** Get  the minMargin that an SVM can return */
  public double getMinMargin() {
    return m_minMargin;
  }

  /** Set whether min/max margins are determined automatically */
  public void setAutoBounds(boolean autoBounds) {
    m_autoBounds = autoBounds;
  }

  /** Get whether min/max margins are determined automatically */
  public boolean getAutoBounds() {
    return m_autoBounds;
  }
  
  /**
   * Returns a description of this classifier.
   *
   * @return a description of this classifier as a string.
   */
  public String toString() {

    if (m_train == null) {
      return "SVMlight: No model built yet.";
    }
    String result = "SVM-light classifier\n";

    return result;
  }


  /** Set the path for the temporary files
   * @param tempDirPath a full path to the temporary directory
   */
  public void setTempDirPath(String tempDirPath) {
    m_tempDirPath = tempDirPath;
  }

  /** Get the path for the temporary files
   * @returns a full path to the temporary directory
   */
  public String getTempDirPath() {
    return m_tempDirPath;
  }

  /** Set the path for the binary files
   * @param tempDirPath a full path to the directory where SVMlight binary files are
   */
  public void setBinPath(String binPath) {
    m_binPath = binPath;
  }

  /** Get the path for the binaries
   * @returns a full path to the binaries directory
   */
  public String getBinPath() {
    return m_binPath;
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

  /**
   * Main method for testing this class.
   *
   * @param argv should contain command line options (see setOptions)
   */
  public static void main(String [] argv) {

    try {
      System.out.println(Evaluation.evaluateModel(new SVMlight(), argv));
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getMessage());
    }
  }
}
