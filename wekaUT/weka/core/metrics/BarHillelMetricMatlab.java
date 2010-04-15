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
 *    BarHillelMetricMatlab.java
 *    Copyright (C) 2002 Sugato Basu
 *
 */

package weka.core.metrics;

import weka.core.*;
import java.util.*;
import java.io.*;

/**
 * Class for performing RCA according to Bar-Hillel's algorithm. <p>
 *
 *
 * @author Sugato Basu
 * @version $Revision: 1.3 $
 */

public class BarHillelMetricMatlab extends OfflineLearnableMetric
  implements OptionHandler {

  /** number of instances used to train metric */
  int m_numInstances = -1;

  /** number of attributes in each instance */
  int m_numAttribs = -1;

  /** instances used to train metric */
  Instances m_trainInstances = null;

  /** chunklet assignments of instances */
  int [] m_chunkletAssignments = null;

  /** full matrix returned by Matlab code */
  protected double [][] m_attrMatrix = null;

  /** We can have different ways of converting from distance to similarity  */
  public static final int CONVERSION_LAPLACIAN = 1;
  public static final int CONVERSION_UNIT = 2;
  public static final int CONVERSION_EXPONENTIAL = 4;
  public static final Tag[] TAGS_CONVERSION = {
    new Tag(CONVERSION_UNIT, "similarity = 1-distance"),
    new Tag(CONVERSION_LAPLACIAN, "similarity=1/(1+distance)"),
    new Tag(CONVERSION_EXPONENTIAL, "similarity=exp(-distance)")
      };
  /** The method of converting, by default laplacian */
  protected int m_conversionType = CONVERSION_LAPLACIAN;

  /** Name of the Matlab program file that computes RCA */
  protected String m_RCAMFile = new String("/var/local/RCAmetric.m");

  /** A timestamp suffix for matching vectors with attributes */
  String m_timestamp = null;

  /** Name of the file where attribute names will be stored */
  //  String m_rcaAttributeMatrixFilename = null;
  //  String m_rcaAttributeMatrixFilenameBase = new String("/var/local/RCAattributeMatrix");
  String m_rcaAttributeMatrixFilename = new String("/var/local/RCAattributeMatrix");
  
  /** Name of the file where dataMatrix will be stored */
  public String m_dataFilename = new String("/var/local/RCAdataMatrix");
  
  /** Name of the file where chunklet assignments will be stored */
  //  public String m_chunkletAssignmentFilename = null;
  //  public String m_chunkletAssignmentFilenameBase = new String("/var/local/RCAchunkletAssignment");
  public String m_chunkletAssignmentFilename = new String("/var/local/RCAchunkletAssignment");

  
  /**
   * Create a new metric.
   * @param numAttributes the number of attributes that the metric will work on
   */ 
  public BarHillelMetricMatlab(int numAttributes) throws Exception {
    buildMetric(numAttributes);
  }

  /** Create a default new metric */
  public BarHillelMetricMatlab() {
  } 
   
  /**
   * Creates a new metric which takes specified attributes.
   *
   * @param _attrIdxs An array containing attribute indeces that will
   * be used in the metric
   */
  public BarHillelMetricMatlab(int[] _attrIdxs) throws Exception {
    setAttrIdxs(_attrIdxs);
    buildMetric(_attrIdxs.length);	
  }

  /**
   * Reset all values that have been learned
   */
  public void resetMetric() throws Exception {
    m_trained = false;
    if (m_attrMatrix != null) { 
      for (int i = 0; i < m_attrMatrix.length; i++) {
	for (int j = 0; j < m_attrMatrix.length; j++) {
	  if (i == j) {
	    m_attrMatrix[i][j] = 1; 
	  } else {
	    m_attrMatrix[i][j] = 0;
	  }
	}
      }
    }
  }

  /**
   * Generates a new Metric. Has to initialize all fields of the metric
   * with default values.
   *
   * @param numAttributes the number of attributes that the metric will work on
   * @exception Exception if the distance metric has not been
   * generated successfully.
   */
  public void buildMetric(int numAttributes) throws Exception {
    m_numAttributes = numAttributes;
    m_attrMatrix = new double[numAttributes][numAttributes];
    m_attrIdxs = new int[numAttributes];
    for (int i = 0; i < numAttributes; i++) {
      for (int j = 0; j < numAttributes; j++) {
	if (i == j) {
	  m_attrMatrix[i][j] = 1; 
	} else {
	  m_attrMatrix[i][j] = 0;
	}
      }
      m_attrIdxs[i] = i;
    }
  }

    
  /**
   * Generates a new Metric. Has to initialize all fields of the metric
   * with default values
   *
   * @param options an array of options suitable for passing to setOptions.
   * May be null. 
   * @exception Exception if the distance metric has not been
   * generated successfully.
   */
  public void buildMetric(int numAttributes, String[] options) throws Exception {
    buildMetric(numAttributes);
  }

  /**
   * Create a new metric for operating on specified instances
   * @param data instances that the metric will be used on
   */
  public  void buildMetric(Instances data) throws Exception {
    m_classIndex = data.classIndex();
    m_numAttributes = data.numAttributes();
    m_trainInstances = data;
    m_numInstances = data.numInstances();

    if (m_classIndex != m_numAttributes-1 && m_classIndex != -1) {
      throw new Exception("Class attribute (" + m_classIndex + ") should be the last attribute!!!");
    }
    if (m_classIndex != -1) {
      m_numAttributes--;
    }
    System.out.println("About to build metric with " + m_numAttributes + " attributes, trainable=" + m_trainable);
    buildMetric(m_numAttributes);
  }
    
  
  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distance(Instance instance1, Instance instance2) throws Exception {
    if (instance1 instanceof SparseInstance && instance2 instanceof SparseInstance) {
      throw new Exception ("Not handled now!!\n\n");
    } else if (instance1 instanceof SparseInstance) {
      throw new Exception ("Not handled now!!\n\n");
    }  else if (instance2 instanceof SparseInstance) {
      throw new Exception ("Not handled now!!\n\n");
    } else {
      // distance is (x-y)^T*A*(x-y)
      // RETURNS SQUARE DISTANCE FOR EFFICIENCY!!
      double [] values1 = instance1.toDoubleArray();
      double [] values2 = instance2.toDoubleArray();
      double distance = 0;
      for (int col = 0; col < m_numAttribs; col++) {
	double innerProduct = 0;
	for (int row = 0; row < m_numAttribs; row++) {
	  if (col == m_classIndex || row == m_classIndex) {
	    continue;
	  }
	  innerProduct += (values1[row]-values2[row]) * m_attrMatrix[row][col];
	}
	distance += innerProduct * (values1[col]-values2[col]);
      }
      //      distance = Math.sqrt(distance);
      return distance;
    }
  }


  /** Return the penalty contribution - distance*distance */
  public double penalty(Instance instance1,
			Instance instance2) throws Exception {
    double distance = distance(instance1, instance2);
    return distance * distance; 
  }

  /** Return the penalty contribution - distance*distance */
  public double penaltySymmetric(Instance instance1,
			Instance instance2) throws Exception {
    double distance = distance(instance1, instance2);
    return distance * distance; 
  }
  

  /**
   * Set the type of  distance to similarity conversion. Values other
   * than CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL will be ignored
   * 
   * @param type type of the similarity to distance conversion to use
   */
  public void setConversionType(SelectedTag conversionType) {
    if (conversionType.getTags() == TAGS_CONVERSION) {
      m_conversionType = conversionType.getSelectedTag().getID();
    }
  }

  /**
   * return the type of distance to similarity conversion
   * @return one of CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL
   */
  public SelectedTag getConversionType() {
    return new SelectedTag(m_conversionType, TAGS_CONVERSION);
  }

    /** The computation of a metric can be either based on distance, or on similarity
   * @returns true because euclidean metrict fundamentally computes distance
   */
  public boolean isDistanceBased() {
    return true;
  }

  /**
   * Returns a similarity estimate between two instances. Similarity is obtained by
   * inverting the distance value using one of three methods:
   * CONVERSION_LAPLACIAN, CONVERSION_EXPONENTIAL, CONVERSION_UNIT.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarity(Instance instance1, Instance instance2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + distance(instance1, instance2));
    case CONVERSION_UNIT:
      return 2 * (1 - distance(instance1, instance2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-distance(instance1, instance2));
    default:
      throw new Exception ("Unknown distance to similarity conversion method");
    }
  }


  public void buildAttributeMatrix (Instances data, int [] clusterAssignments) throws Exception {

    m_chunkletAssignments = clusterAssignments;
    m_numInstances = data.numInstances();
    m_numAttribs = data.instance(1).numAttributes();

    if (m_timestamp == null) { 
      m_timestamp = getLogTimestamp();
    }
      
//      m_rcaAttributeMatrixFilename = new String(m_rcaAttributeMatrixFilenameBase + m_timestamp);
//      m_chunkletAssignmentFilename = new String(m_chunkletAssignmentFilenameBase + m_timestamp);

    System.out.println("About to run RCA on " + m_numInstances + " instances, each with " + m_numAttribs + " attributes");
    dumpInstances(m_dataFilename);
    dumpChunklets(m_chunkletAssignmentFilename);
    prepareMatlab(m_RCAMFile);
    runMatlab(m_RCAMFile, "/var/local/RCAmatlab.output");
    System.out.println("Done running Matlab code ... now parsing matlab output files");

    m_attrMatrix = readMatrix(m_rcaAttributeMatrixFilename);

    if (m_attrMatrix == null) {
      System.out.println("WARNING!! Could not parse matlab output file!!\n\n");
    }
    else {
      System.out.println("Successfully parsed matlab output file and learned metric");
    }
  }

  /**
   * Read column vectors from a text file
   * @param name file name
   * @return a <code>double[][]</code> value
   * @exception Exception if an error occurs
   * @returns double[][] array corresponding to vectors
   */
  public double[][] readMatrix(String name) throws Exception {
    BufferedReader r = new BufferedReader(new FileReader(name));
        
    double[][] vectors = new double[m_numAttribs][m_numAttribs];
    int i = 0, j = 0;
    String s = null;

    // number of rows -- ignore
    s =  r.readLine();
    // number of columns -- ignore
    s = r.readLine();

//      try {
//        int numRows = (int)Double.parseDouble(s);
//      } catch (Exception e) {
//        System.err.println("Couldn't parse " + s + " as Double");
//      }
    
//      try { 
//        int numCols = (int)Double.parseDouble(s);
//      } catch (Exception e) {
//        System.err.println("Couldn't parse " + s + " as Double");
//      }

    while ((s = r.readLine()) != null) {
      StringTokenizer tokenizer = new StringTokenizer(s);
      while (tokenizer.hasMoreTokens()) {
	String value = tokenizer.nextToken();
	try { 
	  vectors[i][j] = Double.parseDouble(value);
	} catch (Exception e) {
	  System.err.println("Couldn't parse " + value + " as double");
	}
	j++;
	if (j > m_numAttribs) {
	  System.err.println("Too many columns (" + j + " instead of " + m_numAttribs + ") in line: " + s);
	}
      }
      if (j != m_numAttribs) {
	System.err.println("Too few columns (" + j + " instead of " + m_numAttribs + ") in line: " + s);
      }
      j = 0;
      i++;
      if (i > m_numAttribs) {
	System.err.println("Too many rows: " + i + ", expecting " + m_numAttribs + " attributes");
      }
    }
    if (i != m_numAttribs) {
      System.err.println("Too few rows: " + i + " expecting " + m_numAttribs + " attributes");
    }
    return vectors;
  }


  /**
   * Dump data matrix into a file
   */
  private void dumpInstances(String tempFile) {
    try { 
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(tempFile)));
      for (int k = 0; k < m_numInstances; k++) {
	Instance instance = m_trainInstances.instance(k);
	for (int j = 0; j < m_numAttribs; j++) {
	  writer.print(instance.value(j) + " ");
	}
	writer.println();
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Could not create a temporary file for dumping the data matrix: " + e);
    }
  }

  /**
   * Dump chunklet vector into a file
   */
  private void dumpChunklets (String tempFile) {
    try { 
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(tempFile)));
      for (int k = 0; k < m_chunkletAssignments.length; k++) {
	writer.print(m_chunkletAssignments[k] + " ");
      }
      writer.println();
      writer.close();
    } catch (Exception e) {
      System.err.println("Could not create a temporary file for dumping the data matrix: " + e);
    }
  }


  /** Create matlab m-file for ICA
   * @param filename file where matlab script is created
   */
  public void prepareMatlab(String filename) {
    try{
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(filename)));
      writer.println("addpath /var/local;");
      writer.println("addpath /u/ml/software/metric-learning/rca;");
      writer.println("data = load('" + m_dataFilename + "');");
      writer.println("chunklets = load('" + m_chunkletAssignmentFilename + "');");
      writer.println("[A] = RCA(data,chunklets);");
      writer.println("[AnumRows, AnumCols] = size(A);");
      writer.println("save " + m_rcaAttributeMatrixFilename + " AnumRows AnumCols A -ASCII -DOUBLE");
      writer.close();
    } 
    catch (Exception e) {
      System.err.println("Could not create matlab file: " + e);
    }
  }


  /** Run matlab in command line with a given argument
   * @param inFile file to be input to Matlab
   * @param outFile file where results are stored
   */
  public static void runMatlab(String inFile, String outFile) {
    // call matlab to do the dirty work
    try {
      int exitValue;
      do {
	System.out.println("Starting to run matlab");
	Process proc = Runtime.getRuntime().exec("matlab -nodesktop -nosplash < " + inFile + " > " + outFile);
	exitValue = proc.waitFor();
	if (exitValue != 0) {
	  System.err.println("WARNING!!!!!  Matlab returned exit value 1, trying again in 5 mins!");
	  Thread.sleep(300000);
	}
      } while (exitValue != 0);
      System.out.println("End of running matlab, exitValue = " + exitValue);
    } 
    catch (Exception e) {
      System.err.println("Problems running matlab: " + e);
    }
  } 

  /** Get a timestamp string as a weak uniqueid
   * @returns a timestamp string in the form "mmddhhmmssS"
   */     
  public static String getLogTimestamp() {
    Calendar cal = Calendar.getInstance(TimeZone.getDefault());
    String DATE_FORMAT = "MMddHHmmssS";
    java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat(DATE_FORMAT);
    
    sdf.setTimeZone(TimeZone.getDefault());          
    return (sdf.format(cal.getTime()));
  }

  /**
   * Given a cluster of instances, return the centroid of that cluster
   * @param instances objects belonging to a cluster
   * @param fastMode whether fast mode should be used for SparseInstances
   * @param normalized normalize centroids for SPKMeans
   * @return a centroid instance for the given cluster
   */
  public Instance getCentroidInstance(Instances instances, boolean fastMode, boolean normalized) {
    System.out.println("\n\nWARNING!! Not implemented!!\n\n");
    return null;
  }
  
  /** Get the values of the partial derivates for the metric components
   * for a particular instance pair
   @param instance1 the first instance
   @param instance2 the first instance
   */
  public double[] getGradients(Instance instance1, Instance instance2) throws Exception {
    System.out.println("\n\nWARNING!! Not implemented!!\n\n");
    return null;
  }

  /**
   * Create an instance with features corresponding to components of the two given instances
   * @param instance1 first instance
   * @param instance2 second instance
   */
  public Instance createDiffInstance (Instance instance1, Instance instance2) {
    System.out.println("\n\nWARNING!! Not implemented!!\n\n");
    return null;
  }

  /**
   * Train the distance metric.  A specific metric will take care of
   * its own training either via a metric learner or by itself.
   */
  public void learnMetric(Instances data) throws Exception {
    System.out.println("\n\nWARNING!! Not implemented!!\n\n");
  }

  /**
   * Returns a distance value between two instances. 
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if distance could not be estimated.
   */
  public double distanceNonWeighted(Instance instance1, Instance instance2) throws Exception {
    System.out.println("\n\nWARNING!! Not implemented!!\n\n");
    return -1;
  }    

  /**
   * Returns a similarity estimate between two instances without using the weights.
   * @param instance1 First instance.
   * @param instance2 Second instance.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarityNonWeighted(Instance instance1, Instance instance2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + distanceNonWeighted(instance1, instance2));
    case CONVERSION_UNIT:
      return 2 * (1 - distanceNonWeighted(instance1, instance2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-distanceNonWeighted(instance1, instance2));
    default:
      throw new Exception ("Unknown distance to similarity conversion method");
    }
  }

  /**
   * Gets the current settings of WeightedEuclideanP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [45];
    int current = 0;

    if (m_conversionType == CONVERSION_EXPONENTIAL) {
      options[current++] = "-E";
    } else if (m_conversionType == CONVERSION_UNIT) {
      options[current++] = "-U";
    }
    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -N <br>
   * Normalize the euclidean distance by vectors lengths
   *
   * -E <br>
   * Use exponential conversion from distance to similarity
   * (default laplacian conversion) <p>
   *
   * -U <br>
   * Use unit conversion from similarity to distance (dist=1-sim)
   * (default laplacian conversion) <p>
   *
   * -R <br>
   * The metric is trainable and will be trained using the current MetricLearner
   * (default non-trainable)
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    if (Utils.getFlag('E', options)) {
      setConversionType(new SelectedTag(CONVERSION_EXPONENTIAL, TAGS_CONVERSION));
    } else if (Utils.getFlag('U', options)) {
      setConversionType(new SelectedTag(CONVERSION_UNIT, TAGS_CONVERSION));
    } else {
      setConversionType(new SelectedTag(CONVERSION_LAPLACIAN, TAGS_CONVERSION));
    }

    if (Utils.getFlag('R', options)) {
      setTrainable(Utils.getFlag('R', options));
      setExternal(Utils.getFlag('X', options));
    }      

    Utils.checkForRemainingOptions(options);
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(4);

    newVector.addElement(new Option("\tNormalize the euclidean distance by vectors lengths\n",
				    "N", 0, "-N"));
    newVector.addElement(new Option("\tUse exponential conversion from similarity to distance\n",
				    "E", 0, "-E"));
    newVector.addElement(new Option("\tUse unit conversion from similarity to distance\n",
				    "U", 0, "-U"));
    newVector.addElement(new Option("\tTrain the metric\n",
				    "R", 0, "-R"));
    newVector.addElement(new Option("\tUse the metric learner for similarity calculations(\"external\")",
				    "X", 0, "-X"));
    newVector.addElement(new Option(
	      "\tFull class name of metric learner to use, followed\n"
	      + "\tby scheme options. (required)\n"
	      + "\teg: \"weka.core.metrics.ClassifierMetricLearner -B weka.classifiers.function.SMO\"",
	      "L", 1, "-L <classifier specification>"));
    
    return newVector.elements();
  }


  /**
   * Main method for testing this class
   * @param argv should contain the command line arguments to the
   * evaluator/transformer (see AttributeSelection)
   */
  public static void main(String [] argv) {
  }  
}

