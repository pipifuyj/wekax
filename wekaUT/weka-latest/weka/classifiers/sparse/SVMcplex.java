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
 *    SVMcplex.java
 *    Copyright (C) 2004 Mikhail Bilenko
 *
 */

package weka.classifiers.sparse;

import weka.classifiers.Classifier;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.UpdateableClassifier;
import java.io.*;
import java.util.*;
import weka.core.*;

import ilog.concert.*;
import ilog.cplex.*;

/**
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */
public class SVMcplex extends DistributionClassifier implements OptionHandler {
  /** Kernel to use */
  protected Kernel m_kernel = null;
  /** The size of the cache (a prime number) */
  private int m_cacheSize = 1000003;
  
  /** The training instances used for classification. */
  protected Instances m_train;

  /** Indeces of support vectors */
  protected int [] m_svIndeces = null;

  /** Lagrange multipliers */
  protected double [] m_alphas = null;

  /** The threshold for the minimum value of alpha */
  protected double m_minAlpha = 1e-8; 
  
  /** Class values for support vectors */
  protected int [] m_classVals = null;
 
  /** The thresholds. */
  private double m_b, m_bLow, m_bUp;
  
  /** The indices for m_bLow and m_bUp */
  private int m_iLow, m_iUp;
  
  
  /** Has the SVM been trained */ 
  protected boolean m_svmTrained = false;

  /** Output debugging information */
  protected boolean m_debug = false;

  /** Path to the directory where temporary files will be stored */
  protected String m_tempDirPath = new String("/var/local/tmp/");
  protected File m_tempDirFile = null;

  /** Temp file storing the problem */
  protected String m_lpFilename = null; 

  /** SVM-light predictions are positive or negative margins; to convert
   * to a distribution we need min/max margin values... */ 
  protected double m_maxMargin = -45;
  protected double m_minMargin = 45;
  protected boolean m_autoBounds = false;

  /** Is classification done via temporary files or via a buffer? */
  protected boolean m_bufferedMode = true;
  protected BufferedReader m_procReader = null;
  protected BufferedWriter m_procWriter = null;

  /** In some cases we don't want feature reduction - then an "all-features" example can be added */
  protected boolean m_useAllFeaturesExample = true; 


  /**********************/

  /** verbosity level */
  protected int m_verbosityLevel = 1;

  /** trade-off between training error and margin (default 0 corresponds to [avg. x*x]^-1) */
  protected double m_C = 10;      
  
  /** Kernel type */ 
  public static final int KERNEL_LINEAR = 1;
  public static final int KERNEL_POLYNOMIAL = 2;
  public static final int KERNEL_RBF = 4;
  public static final Tag[] TAGS_KERNEL_TYPE = {
    new Tag(KERNEL_LINEAR, "Linear"),
    new Tag(KERNEL_POLYNOMIAL, "Polynomial (s a*b+c)^d"),
    new Tag(KERNEL_RBF, "Radial basis function exp(-gamma ||a-b||^2)")
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
  public SVMcplex() {
    try { 

    } catch (Exception e) {
      System.out.println("Problem creating CPLEX factory: " + e);
      e.printStackTrace();
    }
  }
  
  /** Take care of closing the SVM-light process before the object is destroyed
   */
  protected void finalize() {
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
    
    int numInstances = instances.numInstances();
    m_train = new Instances(instances, 0, numInstances);

     // Initialize kernel
    switch (m_kernelType) {
    case KERNEL_LINEAR:
      m_kernel = new PolyKernel(m_train, m_cacheSize, 1.0, false);
      break;
    case KERNEL_RBF:
      m_kernel = new RBFKernel(m_train, m_cacheSize, m_gamma);
      break;
    case KERNEL_POLYNOMIAL:
      m_kernel = new PolyKernel(m_train, m_cacheSize, m_d, false);
      break;
    }
    
    // Unlike most Weka classifiers, we are *not* throwing away training
    // instances with missing class, since they may be used for transduction.
    // If it is desired to avoid transduction and throw out unlabeled data,
    // uncomment the following line:
    m_train.deleteWithMissingClass();
    
    // create the Gram matrix and save it
    double[][] Q = new double[numInstances][numInstances];
    for (int i = 0; i < numInstances; i++) {
      for (int j = 0; j <= i; j++) {
	Q[i][j] = Q[j][i] = m_kernel.eval(i, j, m_train.instance(i));
      }
    }

    // save the QP in a CPLEX file
    m_tempDirFile = new File(m_tempDirPath);
    File lpFile = File.createTempFile("cplex", ".lp", m_tempDirFile);
    if (!m_debug) { 
      lpFile.deleteOnExit();
    }
    m_lpFilename = lpFile.getPath();
    PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(lpFile)));

    writer.println("Minimize");
    writer.print(" obj:");
    for (int i = 0; i < numInstances; i++) {
      writer.print("-x" + i + " ");
    }

    writer.print("+ [");
    for (int i = 0; i < numInstances; i++) {
      for (int j = 0; j <= i; j++) {
	if (Q[i][j] > 0) {
	  writer.print(" + " + Q[i][j] + " x" + i + " * x" + j);
	} else if (Q[i][j] < 0){
	  writer.print(Q[i][j] + "x" + i + " * x" + j);
	}
	if (j > 10) writer.println(); // preventing a silly CPLEX buffer overrun
      }
      writer.println();
    }
    writer.println("]");
    writer.println("Subject To");
    writer.print(" c1: ");
    for (int i = 0; i < numInstances; i++) {
      double classVal = m_train.instance(i).classValue();
      if (classVal == 0) {
	writer.print(" -x" + i);
      } else {
	writer.print(" +x" + i);
      } 
    }
    writer.println(" = 0");
    writer.println("Bounds");
    for (int i = 0; i < numInstances-1; i++) {
      writer.println("0 <= x" + i + " <= " + m_C);
    }
    if (m_useAllFeaturesExample) {
      writer.println(m_minAlpha + " <= x" + (numInstances-1) + " <= " + m_C);
    } else {
      writer.println("0 <= x" + (numInstances-1) + " <= " + m_C);
    } 
    writer.println("End");
    writer.close();
        
    // Train the model 
    trainSVMcplex();
    
  }

  
  /** Launch an SVM-light process assuming that the training data has been dumped
   */
  protected void trainSVMcplex() throws Exception {
    IloCplex cplex = new IloCplex();
    cplex.importModel(m_lpFilename);
    
    if ( cplex.solve() ) {
      System.out.println("Solution status = " + cplex.getStatus());
      System.out.println("Solution value  = " + cplex.getObjValue());
      
      IloLPMatrix lp = (IloLPMatrix)cplex.LPMatrixIterator().next();
      double[] x = cplex.getValues(lp);
      int numSVs = 0; 
      for (int j = 0; j < x.length; j++) {
	if (x[j] > m_minAlpha) {
	  numSVs++;
	} 
      }
      
      m_svIndeces = new int[numSVs];
      m_alphas = new double[numSVs];
      m_classVals = new int[numSVs];
      numSVs = 0;
      for (int i = 0; i < x.length; i++) {
	if (x[i] > m_minAlpha) {
	  m_svIndeces[numSVs] = i;
	  m_alphas[numSVs] = x[i];
	  double classVal = m_train.instance(i).classValue();
	  m_classVals[numSVs] = (classVal == 0) ? -1 : 1;

	  // accumulate m_b - TODO - don't need for now
	  if (x[i] < m_C) {
	  } 
	  
	  numSVs++;
	} 
      }

      // Set threshold
      m_bUp = -1; m_bLow = 1; m_b = 0;
      m_b = (m_bLow + m_bUp) / 2.0;
      System.out.println("**** " + numSVs +"/" + x.length + " support vectors;   b=" + m_b);
      if (m_useAllFeaturesExample) { System.out.println("\tallFEx alpha=" + x[x.length-1]);}
      m_svmTrained = true; 
    }
    //    m_cplex.end();
  } 


  /** Launch an SVM-light process and classify a given instance
   * @param instance an instance that must be classified
   */
  protected double classifySVMcplex(Instance instance) throws Exception {
    double prediction = Double.MIN_VALUE;

    for (int i = 0; i < m_svIndeces.length; i++) { 
      prediction += m_classVals[i] * m_alphas[i] * m_kernel.eval(-1, m_svIndeces[i], instance);
    }
    prediction -= m_b;
    
    return prediction;
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
    double margin = classifySVMcplex(instance);
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


    String cString = Utils.getOption('c', options);
    if (cString.length() != 0) {
      setC(Double.parseDouble(cString));
    }

    if (Utils.getFlag('F', options)) {
      setUseAllFeaturesExample(true);
    }

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
    
    options[current++] = "-c";
    options[current++] = "" + m_C;

    if (m_useAllFeaturesExample) { 
      options[current++] = "-F";
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
    default:
      System.err.println("UNKNOWN KERNEL TYPE: " + m_kernelType);
    }

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

  /** Set the trade-off between training error and margin (default 0 corresponds to [avg. x*x]^-1) */
  public void setC(double C) {
    m_C = C;
  }
  /** Get the trade-off between training error and margin (default 0 corresponds to [avg. x*x]^-1) */
  public double getC() {
    return m_C;
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


  /** The useAllFeaturesExample option */
  public void setUseAllFeaturesExample(boolean use) {
    m_useAllFeaturesExample = use;
  }
  public boolean getUseAllFeaturesExample() {
    return m_useAllFeaturesExample;
  } 
  
  /**
   * Returns a description of this classifier.
   *
   * @return a description of this classifier as a string.
   */
  public String toString() {

    if (m_train == null) {
      return "SVMcplex: No model built yet.";
    }
    String result = "SVM-light classifier\n";

    return result;
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
      System.out.println(Evaluation.evaluateModel(new SVMcplex(), argv));
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getMessage());
    }
  }
}
