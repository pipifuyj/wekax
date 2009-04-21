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
import java.io.*;
import java.text.SimpleDateFormat;

import weka.core.*;

/** 
 * MatlabMetricLearner - learns metric parameters by constructing
 * "difference instances" and then learning weights that classify same-class
 * instances as positive, and different-class instances as negative using an
 * external Matlab program.
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */

public class MatlabMetricLearner extends  MetricLearner implements Serializable {

  /** Matlab program  that is used for learning metric weights */
  protected String m_scriptFilename  = new String("/tmp/matlab1.m");

  /** Name of the temporary file where the matrix representing the same-class diff. instances is going to be */
  protected String m_posMatrixFilename = new String("/tmp/posMatrix.txt");

  /** Name of the temporary file where the matrix representing the diff-class diff. instances is going to be */
  protected String m_negMatrixFilename = new String("/tmp/negMatrix.txt");

  /** Name of the temporary file where the weights will be stored by Matlab after calculation */
  protected String m_weightsFilename = new String("/tmp/weights.txt");

  /** Debugging output */
  protected boolean m_debug = true;

  /** Create a new matlab metric learner
   */
  public MatlabMetricLearner() {
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
    if (instances.classIndex() < 0) {	
      return;
    }

    // First, create positive and negative diff-instances
    ArrayList[] diffInstanceLists = createDiffInstanceLists(instances, metric,
							    metric.getNumPosDiffInstances(), metric.getPosNegDiffInstanceRatio());
    ArrayList posDiffInstanceList = diffInstanceLists[0];
    ArrayList negDiffInstanceList = diffInstanceLists[1];

    prepareMatlabScript();
    dumpInstanceList(posDiffInstanceList, m_posMatrixFilename);
    dumpInstanceList(negDiffInstanceList, m_negMatrixFilename);
    runMatlab(m_scriptFilename, "matlab.out");

    double[] coefficients = readVector(m_weightsFilename);
    if (m_debug) System.out.println(getTimestamp() + " Read " + coefficients.length + " coefficients");
    for (int i = 0; i < coefficients.length; i++) {
      //      coefficients[i] = (coefficients[i]+1)/2;
    } 
    metric.setWeights(coefficients);
  }
  
  /** Create matlab m-file for PCA
   * @param filename file where matlab script is created
   */
  public void prepareMatlabScript() {
    try{
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(m_scriptFilename)));
      //      writer.println("function w = fitMetricWeights()                                               ");
      writer.println("S = load('" + m_posMatrixFilename + "');                                      ");
      writer.println("D = load('" + m_negMatrixFilename + "');                                      ");
      writer.println("[mD,n] = size(D);                                                             ");
      writer.println("[mS,n] = size(S);                                                             ");
      writer.println("");
      writer.println("lb = zeros(n, 1);                                                             ");
      writer.println("ub = ones(n, 1);                                                              ");
      writer.println("x0 = ones(n, 1)/sqrt(n);                                                              ");
      writer.println("");
      writer.println("b = 2* norm(S*x0)/mS * ones(mD, 1);                                              ");
      writer.println("w = fmincon(inline('1/norm(S*x)', 'x', 'S'), x0, D, b, [], [], lb, ub, [],[],S);");
      writer.println("w = w/norm(w)");
      writer.println("save " + m_weightsFilename + " w -ASCII -DOUBLE;");
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
  public void runMatlab(String inFile, String outFile) {
    // call matlab to do the dirty work
    try {
      int exitValue;
      do {
	if (m_debug) System.out.println(getTimestamp() + " starting Matlab");
	Process proc = Runtime.getRuntime().exec("matlab -tty < " + inFile + " > " + outFile);
	exitValue = proc.waitFor();
	if (exitValue != 0) {
	  System.err.println(getTimestamp() + " WARNING!!!!!  Matlab returned exit value 1, trying again later!");
	  Thread.sleep(300000);
	}
      } while (exitValue != 0);
      if (m_debug) System.out.println(getTimestamp() + " Matlab done");
    } catch (Exception e) {
      System.err.println("Problems running matlab: " + e);
    }
  } 


  /**
   * Gets a string containing current date and time.
   *
   * @return a string containing the date and time.
   */
  protected static String getTimestamp() {

    return (new SimpleDateFormat("HH:mm:ss:")).format(new Date());
  }

     /** Read a column vector from a text file
   * @param name file name
   * @returns double[] array corresponding to a vector
   */
  public double[] readVector(String name) throws Exception {
     BufferedReader r = new BufferedReader(new FileReader(name));
     int numAttributes = -1;
     
     ArrayList vectorList = new ArrayList();
     String s;
     while ((s = r.readLine()) != null) {
       try { 
	 vectorList.add(new Double(s));
       } catch (Exception e) {
	 System.err.println("Couldn't parse " + s + " as double");
       }
     }
     int length = vectorList.size();
     double [] vector = new double[length];
     for (int i = 0; i < length; i++) {
       vector[i] = ((Double) vectorList.get(i)).doubleValue();
     } 
     return vector;
  }

  /** Dump a list of instances as a matrix of attribute values
   * @param instanceList a list of instances
   * @param filename name of the file where the matrix is saved
   */
  public void dumpInstanceList(ArrayList instanceList, String filename) {
    try { 
      PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(filename)));
      int numInstances = instanceList.size();
      for (int i = 0; i < numInstances; i++) {
	Instance instance = (Instance) instanceList.get(i);
	int numAttributes = instance.numAttributes();
	int classIdx = instance.classIndex();
	for (int j = 0; j < numAttributes; j++) {
	  if (j != classIdx) {
	    writer.print(instance.value(j) + " ");
	  }
	}
	writer.println();
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Could not create a temporary file for dumping the instance list: " + e);
    }
  }

  /**
   * Use Matlab for an estimation of similarity
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns sim an approximate similarity obtained from the classifier
   */
  public double getSimilarity(Instance instance1, Instance instance2) throws Exception{
    throw new Exception("MatlabMetricLearner cannot be used as an external distance metric!");
  }

  /**
   * Use Matlab for an estimation of distance
   * @param instance1 first instance of a pair
   * @param instance2 second instance of a pair
   * @returns sim an approximate distance obtained from the classifier
   */
  public double getDistance(Instance instance1, Instance instance2) throws Exception{
    throw new Exception("MatlabMetricLearner cannot be used as an external distance metric!");
  }
}





