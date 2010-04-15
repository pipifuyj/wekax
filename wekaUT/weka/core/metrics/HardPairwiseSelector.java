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
 *    HardPairwiseSelector.java
 *    Copyright (C) 2002 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import java.util.*;
import java.io.Serializable;

import weka.core.*;

/** 
 *  HardPairwiseSelector class.  Given a metric and training data,
 * create a set of "difficult" diff-class instance pairs that correspond to metric training data
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.3 $
 */

public class HardPairwiseSelector extends PairwiseSelector implements Serializable, OptionHandler {

  public static final int PAIRS_RANDOM = 1;
  public static final int PAIRS_HARDEST = 2;
  public static final int PAIRS_EASIEST = 4;
  public static final int PAIRS_INTERVAL = 8;
  public static final Tag[] TAGS_PAIR_SELECTION_MODE = {
    new Tag(PAIRS_RANDOM, "Random pairs"),
    new Tag(PAIRS_HARDEST, "Hardest pairs"),
    new Tag(PAIRS_EASIEST, "Easiest pairs"),
    new Tag(PAIRS_INTERVAL, "Pairs in a percentile range")
  };
  protected int m_positivesMode = PAIRS_RANDOM;
  protected int m_negativesMode = PAIRS_RANDOM;


  /** We will need this reverse comparator class to get hardest pairs (those with the largest distance */
  public class ReverseComparator implements Comparator {
    public int compare(Object o1, Object o2) {
      Comparable c = (Comparable) o1;
      return -1 * c.compareTo(o2);
    }
  }


  /** A default constructor */
  public HardPairwiseSelector() {
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
    TreeSet posPairSet = null;
    TreeSet negPairSet = null;
    double [] posPairDistances = null;
    double [] negPairDistances = null;
    Iterator iterator = null;
    int numActualPositives = 0, numActualNegatives = 0;

    // INITIALIZE
    initSelector(instances);
    System.out.println("m_numPotentialPositives=" + m_numPotentialPositives + "\tm_numPotentialNegatives=" + m_numPotentialNegatives);

    // SELECT POSITIVE PAIRS
    switch (m_positivesMode) {
    case PAIRS_EASIEST:
      posPairSet = new TreeSet();
      posPairDistances = populatePositivePairSet(metric, posPairSet);
      pairList = getUniquePairs(posPairSet, metric, numPosPairs);
      break;

    case PAIRS_HARDEST:
      posPairSet = new TreeSet(new ReverseComparator());
      posPairDistances = populatePositivePairSet(metric, posPairSet);
      pairList = getUniquePairs(posPairSet, metric, numPosPairs);
      break;

    case PAIRS_RANDOM:
      // go through lists of instances for each class and create a list of *all* positive pairs
      ArrayList posPairList = new ArrayList();
      iterator = m_classInstanceMap.values().iterator();
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
      } else { // if we have enough potential pairs, sample randomly with replacement
	Random random = new Random();
	for (int i = 0; i < numPosPairs; i++) {
	  int idx = random.nextInt(posPairList.size());
	  TrainingPair pair = (TrainingPair) posPairList.remove(idx);
	  pairList.add(pair);
	}
      }
      break;

    case PAIRS_INTERVAL:
      System.err.println("TODO PAIRS_INTERVAL!!!");
      break;

    default:
      throw new Exception("Unknown method for selecting positive pairs: " + m_positivesMode);
    }
    numActualPositives = pairList.size();
    

    // SELECT NEGATIVE PAIRS
    switch (m_negativesMode) {
    case PAIRS_EASIEST:
      // Create a map with *all* negatives
      negPairSet = new TreeSet(new ReverseComparator());
      negPairDistances = populateNegativePairSet(metric, negPairSet);
      pairList.addAll(getUniquePairs(negPairSet, metric, numNegPairs));

    case PAIRS_HARDEST:
      negPairSet = new TreeSet();
      negPairDistances = populateNegativePairSet(metric, negPairSet);
      pairList.addAll(getUniquePairs(negPairSet, metric, numNegPairs));
      break;

    case PAIRS_RANDOM:         // create all negative pairs and sample randomly
      ArrayList negPairList = new ArrayList();

      // go through lists of instances for each class
      for (int i = 0; i < m_classValueList.size(); i++) {
	ArrayList instanceList1 = (ArrayList) m_classInstanceMap.get(m_classValueList.get(i));
	for (int j = 0; j < instanceList1.size(); j++) {
	  Instance instance1 = (Instance) instanceList1.get(j);
	  // create all pairs from other clusters with this instance
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
      } else { // if we have enough potential pairs, randomly sample with replacement
	Random random = new Random();
	for (int i = 0; i < numNegPairs; i++) {
	  int idx = random.nextInt(negPairList.size());
	  TrainingPair pair = (TrainingPair) negPairList.remove(idx);
	  pairList.add(pair);
	}
      }
      break;

    case PAIRS_INTERVAL:
       System.err.println("TODO PAIRS_INTERVAL!!!");
      break;

    default:
      throw new Exception("Unknown method for selecting positive pairs: " + m_positivesMode);
    }
    numActualNegatives = pairList.size() - numActualPositives;

    System.out.println();
    System.out.println("POSITIVES:  requested=" + numPosPairs + "\tpossible=" + m_numPotentialPositives +
		       "\tactual=" + numActualPositives);
    System.out.println("NEGATIVES:  requested=" + numNegPairs + "\tpossible=" + m_numPotentialNegatives +
		       "\tactual=" + numActualNegatives);
    return pairList;
  }


  /** This helper method goes through a TreeSet containing sorted TrainingPairs
   * and returns a list of unique pairs
   * @param pairSet a sorted set of TrainingPair's
   * @param metric the metric that is used for creating DiffInstance's
   * @param numPairs the number of desired pairs
   * @return a list with training pairs
   */      
  protected ArrayList getUniquePairs(TreeSet pairSet, Metric metric, int numPairs) {
    ArrayList pairList = new ArrayList();
    HashMap checksumMap = new HashMap();
    Iterator iterator = pairSet.iterator();
    for (int i = 0; iterator.hasNext() && i < numPairs; i++) {
      TrainingPair pair = (TrainingPair) iterator.next();
      if (metric instanceof LearnableMetric) {
	Instance diffInstance = ((LearnableMetric)metric).createDiffInstance(pair.instance1, pair.instance2);
	double checksum = 0;
	for (int j = 0; j < diffInstance.numValues(); j++) {
	  checksum += j*17 * diffInstance.value(j);
	}
	// round off to help with machine precision errors
	checksum = (float) checksum;

	// if this checksum was encountered before, get a list of instances
	// that have this checksum, and check if any of them are dupes of this one
	if (checksumMap.containsKey(new Double(checksum))) {
	  ArrayList checksumList = (ArrayList) checksumMap.get(new Double(checksum));
	  System.out.println("Collision for " + checksum + ": " + checksumList.size());
	  boolean unique = true;
	  for (int k = 0; k < checksumList.size() && unique; k++) {
	    Instance nextDiffInstance = (Instance) checksumList.get(k);
	    unique = false;
	    for (int l = 0; l < nextDiffInstance.numValues() && !unique; l++) {
	      if (((float)nextDiffInstance.value(l)) != ((float)diffInstance.value(l))) {
		unique = true;
	      } 
	    }
	    if (!unique) {
	      // This is a dupe!
	      System.out.println("Dupe!");
	      i--;
	      break;
	    } 
	  }
	  if (unique) {
	    pairList.add(pair);
	    checksumList.add(diffInstance);
	  }
	} else {  // this checksum has not been encountered before
	  pairList.add(pair);
	  ArrayList checksumList = new ArrayList();
	  checksumList.add(diffInstance);
	  checksumMap.put(new Double(checksum), checksumList);
	}
      } else { // this is not a LearnableMetric
	pairList.add(pair);
      }
    }
    return pairList;
  }
      



    

  /** Add a pair to the set so that there are no collisions
   * @param set a set to which a new pair should be added
   * @param pair a new pair that is to be added; value is the distance between the instances
   * @return the unique value of the distance (possibly perturbed) with which the pair was added
   */
  protected double addUniquePair(TreeSet set, TrainingPair pair) {
    Random random = new Random();
    double epsilon = 0.00001;
    int counter = 0;
    while (set.contains(pair)) {
      double perturbation;
      if (pair.value == 0) {
	perturbation = Double.MIN_VALUE * random.nextInt(m_numPotentialPositives);
      } else {
	perturbation = pair.value * epsilon * ((random.nextDouble() > 0.5) ? 1 : -1);
      }
      pair.value += perturbation;
      counter++;
      if (counter % 10 == 0) {
	epsilon *= 10;
      }
    }
    set.add(pair);
    return pair.value;
  }

  /** Populate a treeset with all positive TrainingPair's
   * @param metric a metric that will be used to calculate distance
   * @param pairSet an empty set that will be populated
   * @return an array with distance values of the created pairs
   */
  protected double[] populatePositivePairSet(Metric metric, TreeSet pairSet) throws Exception {
    // Create a map with *all* positives
    double [] posPairDistances = new double[m_numPotentialPositives];
    int posCounter = 0;
    // go through lists of instances for each class
    Iterator iterator = m_classInstanceMap.values().iterator();
    while (iterator.hasNext()) {
      ArrayList instanceList = (ArrayList) iterator.next();
      for (int i = 0; i < instanceList.size(); i++) {
	Instance instance1 = (Instance) instanceList.get(i);
	for (int j = i+1; j < instanceList.size(); j++) {
	  Instance instance2 = (Instance) instanceList.get(j);
	  TrainingPair pair = new TrainingPair(instance1, instance2, true, metric.distance(instance1, instance2));
	  // add the pair to the set
	  posPairDistances[posCounter++] = addUniquePair(pairSet, pair);
	} 
      } 
    }
    return posPairDistances;
  }


  /** Populate a treeset with all negative TrainingPair's
   * @param metric a metric that will be used to calculate distance
   * @param pairSet an empty set that will be populated
   * @return an array with distance values of the created pairs
   */
  protected double[] populateNegativePairSet(Metric metric, TreeSet pairSet) throws Exception {
    double [] negPairDistances = new double[m_numPotentialNegatives];
    int negCounter = 0;
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
	    negPairDistances[negCounter++] = addUniquePair(pairSet, pair);
	  }
	}
      }
    }
    return negPairDistances;
  }

  
  /** Given a set, return a TreeSet whose items are accessed in descending order
   * @param set any set containing Comparable objects
   * @return a new ordered set with those objects in reverse order
   */
  public TreeSet reverseCopy(Set set) {
    TreeSet reverseSet = new TreeSet(new ReverseComparator());
    reverseSet.addAll(set);
    return reverseSet;
  }


  /** Set the selection mode for positives
   * @param mode selection mode
   */
  public void setPositivesMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_PAIR_SELECTION_MODE) {
      m_positivesMode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the selection mode for positives
   * @return one of the selection modes
   */
  public SelectedTag getPositivesMode() {
    return new SelectedTag(m_positivesMode, TAGS_PAIR_SELECTION_MODE);
  }


    /** Set the selection mode for negatives
   * @param mode selection mode
   */
  public void setNegativesMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_PAIR_SELECTION_MODE) {
      m_negativesMode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the selection mode for negatives
   * @return one of the selection modes
   */
  public SelectedTag getNegativesMode() {
    return new SelectedTag(m_negativesMode, TAGS_PAIR_SELECTION_MODE);
  }
  

   /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [5];
    int current = 0;

    options[current++] = "-P";
    switch(m_positivesMode) {
    case PAIRS_RANDOM:
	options[current++] = "-r";
	break;
    case PAIRS_HARDEST:
      options[current++] = "-h";
	break;
    case PAIRS_EASIEST:
	options[current++] = "-e";
	break;
    case PAIRS_INTERVAL:
      options[current++] = "-i";
      break;
    }

    options[current++] = "-N";
    switch(m_negativesMode) {
    case PAIRS_RANDOM:
	options[current++] = "-r";
	break;
    case PAIRS_HARDEST:
      options[current++] = "-h";
	break;
    case PAIRS_EASIEST:
	options[current++] = "-e";
	break;
    case PAIRS_INTERVAL:
      options[current++] = "-i";
      break;
    }
    
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

  /**
   * get an array of numIdxs random indeces out of n possible values.
   * if the number of requested indeces is larger then maxIdx, returns
   * maxIdx permuted values
   * @param maxIdx - the maximum index of the set
   * @param numIdxs number of indexes to return
   * @return an array of indexes
   */
  public static int[] randomSubset(int numIdxs, int maxIdx) {
    Random r = new Random(maxIdx + numIdxs);
    int[] indexes = new int[maxIdx];

    for (int i = 0; i < maxIdx; i++) {
      indexes[i] = i;
    }

    // permute the indeces randomly
    for (int i = 0; i < maxIdx; i++) {
      int idx = r.nextInt (maxIdx - i);
      int temp = indexes[i + idx];
      indexes[i + idx] = indexes[i];
      indexes[i] = temp;
    }
    int []returnIdxs = new int[Math.min(numIdxs,maxIdx)];
    for (int i = 0; i < returnIdxs.length; i++) {
      returnIdxs[i] = indexes[i];
    }
    return returnIdxs;
  }

}

