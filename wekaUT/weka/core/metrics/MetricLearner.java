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

import java.util.ArrayList;
import java.io.Serializable;

import weka.core.*;

/** 
 * Abstract MetricLearner interface.  Given a metric and training data,
 * learn the metric's parameters and set them. 
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.2 $
 */

public abstract class MetricLearner {

  /**
   * Train a given metric using given training instances
   *
   * @param metric the metric to train
   * @param instances data to train the metric on
   * @exception Exception if training has gone bad.
   */
  public abstract void trainMetric(LearnableMetric metric, Instances instances) throws Exception;

  /**
   * Use the metricLearner's internal model for an estimation of similarity, e.g.
   * Classifiers can output an approximate margin...
   */
  public abstract double getSimilarity(Instance instance1, Instance instance2) throws Exception;


  /**
   * Use the metricLearner's internal model for an estimation of distance, e.g.
   * Classifiers can output an approximate margin...
   */
  public abstract double getDistance(Instance instance1, Instance instance2) throws Exception;

  
  /** Create two lists:  one of diff-instances belonging to same class,
   * another of diff-instances belonging to different classes.
   *
   * @param instances a set of training instances
   * @param metric the metric to train
   * @returns an array of two lists: first of diff-instances of same class, second of
   * diff-instances of different classes
   */
  protected ArrayList[] createDiffInstanceLists(Instances instances, LearnableMetric metric,
						int numPosDiffInstances, double posNegDiffInstanceRatio) {
    ArrayList[] lists = new ArrayList[2];
    ArrayList diffInstanceList = new ArrayList();

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

    // Create the desired number of positive instances
    int numClasses = classInstanceMap.size();
    Random random = new Random();
    lists[0] = new ArrayList();
    for (int i = 0; i < numPosDiffInstances; i++) {
      // select a random class... TODO: probability must be proportional to the number of instances
      int class1 = random.nextInt(numClasses);
      ArrayList list = (ArrayList) classInstanceMap.get(classValueList.get(class1));
      Instance instance1 = (Instance) list.get(random.nextInt(list.size()));
      Instance instance2 = (Instance) list.get(random.nextInt(list.size()));
      Instance diffInstance = metric.createDiffInstance(instance1, instance2);
      lists[0].add(diffInstance);
    }

    // Create negative diff-instances
    if (numClasses > 1) {
      random = new Random();
      lists[1] = new ArrayList();
      for (int i = 0; i < ((double) lists[0].size()) * posNegDiffInstanceRatio; i++) {
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
	Instance diffInstance = metric.createDiffInstance(instance1, instance2);
	lists[1].add(diffInstance);
      }
    }

    return lists;
  }

  /**
   * Given two ArrayList of pairs of same-class and different-class diff-instances,
   * create an Instances dataset of DiffInstances
   *
   * @param posDiffInstanceList list of diff-instances from same class
   * @param negDiffInstanceList list of diff-instances from different classes
   * @returns a dataset of diff-instances
   */
  protected Instances createDiffInstances (ArrayList posDiffInstanceList,
					   ArrayList negDiffInstanceList) {
    int numPosDiffInstances = (posDiffInstanceList != null) ? posDiffInstanceList.size() : 0;
    int numNegDiffInstances = (negDiffInstanceList != null) ? negDiffInstanceList.size() : 0;
    int numDiffInstances = numPosDiffInstances + numNegDiffInstances;

    System.out.println("Training on " + numPosDiffInstances +
		       " positive and " + numNegDiffInstances + " negative examples");
    if (numDiffInstances == 0) {
      return null;
    }
    // get attribute from a sample diff instance
    Instance sampleInstance = (numPosDiffInstances > 0) ? (Instance)(posDiffInstanceList.get(0)) :
      (Instance)(negDiffInstanceList.get(0));
    weka.core.FastVector attrInfoFVector = getAttrInfoForDiffInstance(sampleInstance);
    Instances diffInstances = new Instances ("DiffInstances", attrInfoFVector, numDiffInstances);
	
    diffInstances.setClassIndex(diffInstances.numAttributes() - 1);

    for (int i = 0; i < numPosDiffInstances; i++) {
      Instance diffInstance = (Instance)posDiffInstanceList.get(i); 
      diffInstance.setDataset(diffInstances);
      diffInstance.setClassValue("pos");
      diffInstances.add(diffInstance);
    }

    for (int i = 0; i < numNegDiffInstances; i++) {
      Instance diffInstance = (Instance)negDiffInstanceList.get(i); 
      diffInstance.setDataset(diffInstances);
      diffInstance.setClassValue("neg");
      diffInstances.add(diffInstance);
    }
    return diffInstances;
  }


    /**
   * Given an ArrayList of TrainingPair's of same-class and different-class diff-instances,
   * create an Instances dataset of DiffInstances
   *
   * @param pairList list of TrainingPair's of instances
   * @param metric a metric that will create the diffInstances
   * @returns a dataset of diff-instances
   */
  protected Instances createDiffInstances (ArrayList pairList, LearnableMetric metric) {
    int numPairs = pairList.size();
    
    // get attribute from a sample diff instance
    Instance sampleInstance = ((TrainingPair)pairList.get(0)).instance1;
    weka.core.FastVector attrInfoFVector = getAttrInfoForDiffInstance(sampleInstance);
    Instances diffInstances = new Instances ("DiffInstances", attrInfoFVector, numPairs);
    
    diffInstances.setClassIndex(diffInstances.numAttributes() - 1);

    for (int i = 0; i < numPairs; i++) {
      TrainingPair pair = (TrainingPair) pairList.get(i);
      Instance diffInstance = metric.createDiffInstance(pair.instance1, pair.instance2);
      diffInstance.setDataset(diffInstances);
      if (pair.positive) {
	diffInstance.setClassValue("pos");
      } else {
	diffInstance.setClassValue("neg");
      }
      diffInstances.add(diffInstance);
    }
    return diffInstances;
  }
  

  /**
   * Given an instance, return a FastVector of attributes.
   * This really should have been in weka.core.Instance... argh.
   * We skip the class index and add a new class Attribute
   *
   * @param instance Instance from which to extract attributes
   */
  protected FastVector getAttrInfoForDiffInstance (Instance instance) {
    FastVector attrFVector = new FastVector();

    int classIndex = (instance.dataset() != null) ? instance.classIndex() : -1;
	
    for (int i = 0; i < instance.numAttributes(); i++) {
      Attribute a = instance.attribute(i);
      // skip the class attribute
      if (i != classIndex) {
	attrFVector.addElement(a);
      }
    }
    // add a difference class attribute
    String s1 = "pos";
    String s2 = "neg";
    weka.core.FastVector classValueFVector = new FastVector();
    classValueFVector.addElement(s1);
    classValueFVector.addElement(s2);
    Attribute classAttr = new Attribute("__class__", classValueFVector);
    attrFVector.addElement(classAttr);

    return attrFVector;
  }


  /**
   * Creates a new instance of a metric learner given it's class name and
   * (optional) arguments to pass to it's setOptions method. If the
   * classifier implements OptionHandler and the options parameter is
   * non-null, the classifier will have it's options set.
   *
   * @param metricLearnerName the fully qualified class name of the metric learner
   * @param options an array of options suitable for passing to setOptions. May
   * be null.
   * @return the newly created metric learner, ready for use.
   * @exception Exception if the metric learner name is invalid, or the options
   * supplied are not acceptable to the metric learner
   */
  public static MetricLearner forName(String metricLearnerName,
				      String [] options) throws Exception {
    System.out.println("Instantiating a metric learner: " + metricLearnerName +
		       " with options: " + weka.classifiers.sparse.IBkMetric.concatStringArray(options));
    MetricLearner m =  (MetricLearner)Utils.forName(MetricLearner.class,
					metricLearnerName,
					options);
    System.out.println("success");
    return m; 
    }


}









