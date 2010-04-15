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
 *    DedupingEvaluation.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package  weka.deduping;

import  java.util.*;
import  java.io.*;
import  weka.core.*;
import  weka.filters.Filter;
import  weka.filters.unsupervised.attribute.Remove;

/**
 * Class for evaluating deduping
 *
 * @author  Mikhail Bilenko
 */
public class DedupingEvaluation {

  /** The number of produced clusters */
  protected int m_numClusters;

  /** Training instances */
  protected Instances m_trainInstances;
  
  /** Test instances */
  protected Instances m_testInstances;

  /** Array for storing the confusion matrix. */
  protected double [][] m_ConfusionMatrix;

  /**
   * Returns a string describing this evaluator
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return " A deduping evaluator that evaluates results of running a "
      + "deduping experiment.";
  }

  /** A default constructor */
  public DedupingEvaluation () {
  }

  /** Train a deduper on the supplied data
   * @param deduper a deduper to train
   * @param data training data
   */
  public void trainDeduper(Deduper deduper, Instances trainingData, Instances testData) throws Exception {
    deduper.buildDeduper(trainingData, testData);
  } 

  /**
   * Evaluates the deduper on a given set of test instances
   *
   * @param clusterer semi-supervised clusterer 
   * @param testInstances set of test instances for evaluation
   * @return a list of arrays containing the basic statistics for each point
   * @exception Exception if model could not be evaluated successfully
   */
  public ArrayList evaluateModel (Deduper deduper, Instances testInstances) throws Exception {
    ArrayList resultList = new ArrayList();
    m_testInstances = testInstances;
    // Run the deduper collecting data
    int numTrueClasses = countPresentClasses(testInstances);
    int numObjects = (int) (0.8 * numTrueClasses);
    System.out.println("testInstances: " + testInstances.numInstances() + " true=" + numTrueClasses + " desired:" + numObjects);
    System.out.println("numClasses=" + testInstances.numClasses());
    deduper.findDuplicates(testInstances, numObjects);
    return deduper.getStatistics();
  }

  /** A helper function that determines how many classes are actually
   * represented in an Instances object
   * @param instances a set of instances
   * @return the number of classes present among the instances
   */
  protected int countPresentClasses(Instances instances) {
    HashSet classValueSet = new HashSet();
    for (int i = 0; i < instances.numInstances(); i++) {
      Instance instance = (Instance) instances.instance(i);
      classValueSet.add(new Double(instance.classValue()));
    }
    return classValueSet.size();
  } 
}
