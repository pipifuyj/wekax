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
 *    RandomPairwiseSelector.java
 *    Copyright (C) 2002 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import java.util.*;
import java.io.Serializable;


import weka.core.*;

/** 
 *  RandomPairwiseSelector class.  Given a metric and training data,
 * create a set of random instance pairs that correspond to metric training data
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */

public class RandomPairwiseSelector extends PairwiseSelector  implements Serializable, OptionHandler {

  /** A default constructor */
  public RandomPairwiseSelector() {
  } 

  /**
   * Provide an array of metric pairs metric using given training instances
   *
   * @param metric the metric to train
   * @param instances data to train the metric on
   * @exception Exception if training has gone bad.
   */
  public ArrayList createPairList(Instances instances, int numPosPairs, int numNegPairs, Metric metric) throws Exception {
    ArrayList pairList = new ArrayList();
    initSelector(instances);

    // go through lists of instances for each class and create a list of *all* positive pairs
    ArrayList posPairList = new ArrayList();
    Iterator iterator = m_classInstanceMap.values().iterator();
    while (iterator.hasNext()) {
      ArrayList instanceList = (ArrayList) iterator.next();
      for (int i = 0; i < instanceList.size(); i++) {
	Instance instance1 = (Instance) instanceList.get(i);
	for (int j = i+1; j < instanceList.size(); j++) {
	  Instance instance2 = (Instance) instanceList.get(j);
	  TrainingPair pair = new TrainingPair(instance1, instance2, true, metric.distance(instance1, instance2));
	  posPairList.add(pair);
	} 
      } 
    }

    // if we have fewer pairs available than requested, return all the ones that were created
    if (posPairList.size() <= numPosPairs) {
      pairList = posPairList;
    } else { // randomly sample with replacement
      Random random = new Random();
      for (int i = 0; i < numPosPairs; i++) {
	int idx = random.nextInt(posPairList.size());
	TrainingPair pair = (TrainingPair) posPairList.remove(idx);
	pairList.add(pair);
      }
    }

    // Analogously, create all negative pairs and sample randomly
    ArrayList negPairList = new ArrayList();
    // go through lists of instances for each class
    for (int i = 0; i < m_classValueList.size(); i++) {
      ArrayList instanceList1 = (ArrayList) m_classInstanceMap.get(m_classValueList.get(i));
      for (int j = 0; j < instanceList1.size(); j++) {
	Instance instance1 = (Instance) instanceList1.get(j);

	for (int k = i+1; k < m_classValueList.size(); k++) {
	  ArrayList instanceList2 = (ArrayList) m_classInstanceMap.get(m_classValueList.get(k));
	  for (int l = 0; l < instanceList2.size(); l++) {
	    Instance instance2 = (Instance) instanceList2.get(l);
	    TrainingPair pair = new TrainingPair(instance1, instance2, false, metric.distance(instance1, instance2));
	    negPairList.add(pair);
	  }
	}
      }
    }

        // if we have fewer pairs available than requested, return all the ones that were created
    if (negPairList.size() <= numNegPairs) {
      pairList.addAll(negPairList);
    } else { // randomly sample with replacement
      Random random = new Random();
      for (int i = 0; i < numNegPairs; i++) {
	int idx = random.nextInt(negPairList.size());
	TrainingPair pair = (TrainingPair) negPairList.remove(idx);
	pairList.add(pair);
      }
    }

    return pairList;
  }

     /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [1];
    int current = 0;

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }
  
  /**
   * Parses a given list of options. Valid options are:<p>
   *
   */
  public void setOptions(String[] options) throws Exception {
  }

    /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
     Vector newVector = new Vector(0);

    return newVector.elements();
  }

}









