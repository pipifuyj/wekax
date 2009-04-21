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
 *    PairwiseSelector.java
 *    Copyright (C) 2002 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import java.util.*;
import java.io.Serializable;

import java.util.ArrayList;

import weka.core.*;

/** 
 * Abstract PairwiseSelector class.  Given a metric and training data,
 * create a set of instance pairs that correspond to metric training data
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.2 $
 */

public abstract class PairwiseSelector {
  /** A hashmap where class attribute values are mapped to lists of instances of that class */
  protected HashMap m_classInstanceMap = null;

  /** A list of classes, each element is the double value of the class attribute */
  protected ArrayList m_classValueList = null;

  /** The number of possible same-class pairs */
  protected int m_numPotentialPositives = 0;

  /** The number of possible different-class pairs */
  protected int m_numPotentialNegatives = 0;

  
  /**
   * Provide an array of metric pairs metric using given training instances
   *
   * @param metric the metric to train
   * @param instances data to train the metric on
   * @exception Exception if training has gone bad.
   */
  public abstract ArrayList createPairList(Instances instances, int numPosPairs, int numNegPairs, Metric metric) throws Exception;

  /** Initialize m_classInstanceMap and m_classValueList using a given set of instances */
  public void initSelector(Instances instances) {
    HashMap sumMap = new HashMap();
    m_classValueList = new ArrayList();
    m_classInstanceMap = new HashMap();
    m_numPotentialPositives = 0;
    m_numPotentialNegatives = 0;

    // go through all instances, hashing them into lists corresponding to each class
    System.out.println("Got " + instances.numInstances());
    Enumeration enum = instances.enumerateInstances();
    int counter = 0;
    while (enum.hasMoreElements()) {
      Instance instance = (Instance) enum.nextElement();
      if (instance.classIsMissing()) {
	System.err.println("Instance has missing class!!!");
	continue;
      }
      Double classValue = new Double(instance.classValue());
      
      // check whether this class has been seen, and get its list of instances
      ArrayList classInstanceList;
      if (m_classInstanceMap.containsKey(classValue)) {
	classInstanceList = (ArrayList) m_classInstanceMap.get(classValue);
      } else {  // create a new list of instances for a previously unseen class
	classInstanceList = new ArrayList();
	m_classInstanceMap.put(classValue, classInstanceList);
	m_classValueList.add(classValue);
      }

      // check that the instance is not a dupe of previously seen instances
      double valueSum = 0;
      for (int i = 0; i < instance.numValues(); i++) {
	valueSum += instance.value(i);
      }

      // prevent duplicate training instances
      if (sumMap.containsKey(new Double(valueSum))) {
	ArrayList sumList = (ArrayList) sumMap.get(new Double(valueSum));
	boolean unique = true;
	for (int i = 0; i < sumList.size() && unique; i++) {
	  Instance nextInstance = (Instance) sumList.get(i);
	  unique = false;
	  for (int j = 0; j < nextInstance.numValues() && !unique; j++) {
	    if (nextInstance.value(j) != instance.value(j)) {
	      unique = true;
	    } 
	  }
	  if (!unique) {
	    // This is a dupe!
	    break;
	  } 
	}
	if (unique) {
	  classInstanceList.add(instance);	  
	}
      } else {
	classInstanceList.add(instance);
	ArrayList sumList = new ArrayList();
	sumList.add(instance);
	sumMap.put(new Double(valueSum), sumList);
      }
    }

    // get the number of potential positive pairs
    Iterator iterator = m_classInstanceMap.values().iterator();
    while (iterator.hasNext()) {
      ArrayList classInstanceList = (ArrayList) iterator.next();
      m_numPotentialPositives += classInstanceList.size() * (classInstanceList.size() - 1) / 2;
      System.out.println(classInstanceList.size() + "\t" + m_numPotentialPositives);
    }
    int numInstances = instances.numInstances();
    m_numPotentialNegatives = numInstances * (numInstances - 1) / 2 - m_numPotentialPositives;
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
  public static PairwiseSelector forName(String pairwiseSelectorName,
				      String [] options) throws Exception {
    System.out.println("Instantiating a pairwise selector: " + pairwiseSelectorName +
		       " with options: " + weka.classifiers.sparse.IBkMetric.concatStringArray(options));
    PairwiseSelector p =  (PairwiseSelector)Utils.forName(PairwiseSelector.class,
					pairwiseSelectorName,
					options);
    System.out.println("success");
    return p; 
    }


}









