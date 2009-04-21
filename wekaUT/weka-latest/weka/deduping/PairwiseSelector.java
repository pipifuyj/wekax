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
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.deduping;

import java.util.*;
import java.io.Serializable;

import weka.core.*;
import weka.deduping.metrics.*;
import weka.deduping.blocking.*;

/** 
 * PairwiseSelector class.  Given a string metric and training data,
 * create a set of instance pairs that correspond to metric training data
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.11 $
 */

public class PairwiseSelector implements OptionHandler, Serializable {
  /** The set of instances used for training */
  protected Instances m_instances = null;
  
  /** A hashmap where true object IDs are mapped to lists of strings of that object */
  protected HashMap m_classInstanceMap = null;

  /** A list of classes, each element is the double value of the class attribute */
  protected ArrayList m_classValueList = null;

  /** A list with all the positive examples as TrainingPair's */
  protected ArrayList m_posPairList = null;

  /** A list with a sufficient pool of negative examples as TrainingPair's */
  protected ArrayList m_negPairList = null;
  
  /** The number of possible same-class pairs */
  protected int m_numPotentialPositives = 0;

  /** The number of possible different-class pairs */
  protected int m_numPotentialNegatives = 0;

  /** Output debugging information */
  protected boolean m_debug = false;

  /** The record pair selection method */

  // positives
  public static final int POS_MODE_RANDOM_RECORDS = 1;
  public static final int POS_MODE_RANDOM_POSITIVES = 2;
  public static final int POS_MODE_STATIC_ACTIVE = 4;
  public static final Tag[] TAGS_POS_MODE = {
    new Tag(POS_MODE_RANDOM_RECORDS, "Random record pairs"),
    new Tag(POS_MODE_RANDOM_POSITIVES, "Random positive pairs"),
    new Tag(POS_MODE_STATIC_ACTIVE, "Static-active positive pairs"),
  };
  protected int m_positivesMode = POS_MODE_RANDOM_POSITIVES;

  // should the rejected positives be spilled into the negatives set?
  protected boolean m_useRejectedPositives = true;

  // negatives
  public static final int NEG_MODE_RANDOM_RECORDS = 1;
  public static final int NEG_MODE_RANDOM_NEGATIVES = 2;
  public static final int NEG_MODE_IMPLICIT_NEGATIVES = 4;
  public static final Tag[] TAGS_NEG_MODE = {
    new Tag(NEG_MODE_RANDOM_RECORDS, "Random record pairs"),
    new Tag(NEG_MODE_RANDOM_NEGATIVES, "Random negative pairs"),
    new Tag(NEG_MODE_IMPLICIT_NEGATIVES, "Implicit negative pairs")
  };
  protected int m_negativesMode = NEG_MODE_RANDOM_NEGATIVES;

  // should falsely selected implicit negatives be spilled into the positives set?
  protected boolean m_useFalseImplicitNegatives = true;


  /** String pair selection method */
  public static final int STRING_PAIRS_RANDOM = 1;
  public static final int STRING_PAIRS_HARDEST = 2;
  public static final int STRING_PAIRS_EASIEST = 4;
  public static final Tag[] TAGS_STRING_PAIR_MODE = {
     new Tag(STRING_PAIRS_RANDOM, "Random string pairs"),
     new Tag(STRING_PAIRS_HARDEST, "Hardest string pairs"),
     new Tag(STRING_PAIRS_EASIEST, "Easiest string pairs")
       };
  protected int m_posStringMode = STRING_PAIRS_RANDOM;
  protected int m_negStringMode = STRING_PAIRS_RANDOM;
  
 
  
  /** The maximum fraction of common tokens that instances can have to
      be included as implicit negatives */
  protected double m_maxImplicitCommonTokenFraction = 0.2;
  
  /** We will need this reverse comparator class to traverse a TreeSet backwards */
  public class ReverseComparator implements Comparator {
    public int compare(Object o1, Object o2) {
      Comparable c = (Comparable) o1;
      return -1 * c.compareTo(o2);
    }
  }


  /** A default constructor */
  public PairwiseSelector() {
  }

  /** Initialize m_classInstanceMap and m_classValueList using a given set of instances
   * @param instances a set of instances from which pair examples will be selected
   */
  public void initSelector(Instances instances) {
    m_instances = instances;
    m_classValueList = new ArrayList();
    m_classInstanceMap = new HashMap();
    m_numPotentialPositives = 0;
    m_numPotentialNegatives = 0;

    // go through all instances, hashing them into lists corresponding to each class
    Enumeration enum = instances.enumerateInstances();
    while (enum.hasMoreElements()) {
      Instance instance = (Instance) enum.nextElement();
      if (instance.classIsMissing()) {
	System.err.println("Instance " + instance + " has missing class!!!");
	continue;
      }
      Double classValue = new Double(instance.classValue());

      // if this class has been seen, add instance to the class's list
      if (m_classInstanceMap.containsKey(classValue)) {
	ArrayList classInstanceList = (ArrayList) m_classInstanceMap.get(classValue);
	classInstanceList.add(instance);
      } else {  // create a new list of instances for a previously unseen class
	ArrayList classInstanceList = new ArrayList();
	classInstanceList.add(instance);
	m_classInstanceMap.put(classValue, classInstanceList);
	m_classValueList.add(classValue);
      }
    }

    // get the number of potential positive pairs
    Iterator iterator = m_classInstanceMap.values().iterator();
    while (iterator.hasNext()) {
      ArrayList classInstanceList = (ArrayList) iterator.next();
      m_numPotentialPositives += classInstanceList.size() * (classInstanceList.size() - 1) / 2;
    }
    int numInstances = instances.numInstances();
    m_numPotentialNegatives = numInstances * (numInstances - 1) / 2 - m_numPotentialPositives;

    createPosPairList();
    createNegPairList();
    System.out.println("m_numPotentialPositives=" + m_numPotentialPositives + "\tm_numPotentialNegatives=" + m_numPotentialNegatives);
  }


  /** Generate a training set of diffInstances. initSelector must have been called earlier
   * to initialize m_posPairList and m_negPairList.      
   * @param attrIdxs indeces of fields that should be utilized
   * @param stringMetrics metrics that should be used on training pairs to generate diffInstances
   * @param numPosPairs the desired number of positive (same-class) diffInstance's
   * @param numNegPairs the desired number of negative (different-class) diffInstance's
   */
  public Instances getInstances(int [] attrIdxs, StringMetric[][] stringMetrics,
				int numPosPairs, int numNegPairs) throws Exception {
    int numActualPositives = 0;
    int numActualNegatives = 0;
    HashMap checksumMap = new HashMap();
    HashSet usedPairSet = new HashSet();
    int numTrainingRecords = m_instances.numInstances();
    double[] checksumCoeffs = new double[stringMetrics.length * stringMetrics[0].length];

    if (m_posPairList == null || m_negPairList == null) {
      throw new Exception("Called PairwiseSelector.getInstances before initalization via initSelector!");
    }

    /*** Create the Instances dataset ***/
    // first, create all the numeric attributes
    FastVector attrInfoVector = new FastVector();
    Random r = new Random(numPosPairs + numNegPairs);
    for (int i = 0; i < stringMetrics.length; i++) {
      for (int j = 0; j < stringMetrics[i].length; j++) { 
	Attribute attr = new Attribute("" + i + "-" + j);
	attrInfoVector.addElement(attr);
	checksumCoeffs[i*stringMetrics[i].length + j] = r.nextDouble();
      }
    }

    // create the class attribute
    FastVector classValues = new FastVector();
    classValues.addElement("pos");
    classValues.addElement("neg");
    Attribute classAttr = new Attribute("class", classValues);
    attrInfoVector.addElement(classAttr);
    // create the dataset and set the class attribute
    Instances instances = new Instances("diffInstances", attrInfoVector, numPosPairs + numNegPairs);
    instances.setClass(classAttr);
    
    /*** Positives selection ***/
    switch (m_positivesMode) {
    case POS_MODE_RANDOM_RECORDS:
      // just pick m_numPosPairs random record pairs
      int numMisfires = 0; 
      for (int i = 0; i < numPosPairs && numMisfires < 1000; i++) {
	InstancePair pair = createRandomTrainInstancePair(usedPairSet, checksumMap); 	  
	Instance trainInstance = createInstance(pair, attrIdxs, stringMetrics);
	if (trainInstance != null && isUniqueInstance(trainInstance, checksumMap, checksumCoeffs)) {
	  instances.add(trainInstance);
	  if (trainInstance.value(trainInstance.numValues()-1) == 0) {
	    numActualPositives++;
	  } else {
	    numActualNegatives++;
	  }
	} else {
	  i--; 
	  numMisfires++;
	}
      }
      break;

    case POS_MODE_RANDOM_POSITIVES:
      // we are sampling from all same-class pairs in the training fold
      // randomize the indeces of positive examples and select the desired number
      numMisfires = 0; 
      int [] posPairIdxs = randomSubset(m_numPotentialPositives, m_numPotentialPositives);
      for (int i = 0; i < posPairIdxs.length && numActualPositives < numPosPairs && numMisfires < 500; i++) {
	Instance posInstance = createInstance((InstancePair) m_posPairList.get(posPairIdxs[i]),
					      attrIdxs, stringMetrics);
	if (posInstance != null && isUniqueInstance(posInstance, checksumMap, checksumCoeffs)) {
	  instances.add(posInstance);
	  numActualPositives++;
	} else {
	  numMisfires++;
	}
      }
      break;

    case POS_MODE_STATIC_ACTIVE:
      Blocking blocker = new Blocking();
      blocker.buildIndex(m_instances);
      InstancePair[] pairs = blocker.getMostSimilarPairs(numPosPairs*2);
      numMisfires = 0; 
      for (int i = 0;
	   (numActualPositives + numActualNegatives) < numPosPairs && i < pairs.length && pairs[i] != null && numMisfires < 500; i++) {
	Instance trainInstance = createInstance(pairs[i], attrIdxs, stringMetrics);
	if (trainInstance != null && isUniqueInstance(trainInstance, checksumMap, checksumCoeffs)) {
	  if (pairs[i].positive == true) { 
	    instances.add(trainInstance);
	    numActualPositives++;
	  } else {
	    if (m_useRejectedPositives) {
	      instances.add(trainInstance);
	      numActualNegatives++;
	    } 
	  }
	} else {
	  numMisfires++;
	}
      }
      System.out.println("After static-active:\t" + numActualPositives + " positives and " +
			 numActualNegatives + " negatives");

      break;
    default:
      throw new Exception("Unknown positive selection mode: " + m_positivesMode);
    }


    
    /*** Negatives selection ***/
    switch (m_negativesMode) {

    case NEG_MODE_RANDOM_RECORDS:
      // just pick m_numNegPairs random record pairs
      int numMisfires = 0; 
      for (int i = 0; i < numNegPairs && numMisfires < 1000; i++) {
	InstancePair pair = createRandomTrainInstancePair(usedPairSet, checksumMap); 	  
	Instance trainInstance = createInstance(pair, attrIdxs, stringMetrics);
	if (trainInstance != null && isUniqueInstance(trainInstance, checksumMap, checksumCoeffs)) {
	  instances.add(trainInstance);
	  if (trainInstance.value(trainInstance.numValues()-1) == 0) {
	    numActualPositives++;
	  } else {
	    numActualNegatives++;
	  }
	} else {
	  i--;
	  numMisfires++;
	}
      }
      break;

    case NEG_MODE_RANDOM_NEGATIVES:
      // we are sampling from all different-class pairs in the training fold
      // randomize the indeces of negative examples and select the desired number
      numMisfires = 0;
      int numUniqueNegatives = 0; 
      int [] negPairIdxs = randomSubset(m_numPotentialNegatives, m_numPotentialNegatives);
      for (int i = 0; i < negPairIdxs.length && numUniqueNegatives < numNegPairs && numMisfires < 1000; i++) {
	Instance negInstance = createInstance((InstancePair) m_negPairList.get(negPairIdxs[i]),
					      attrIdxs, stringMetrics);
	if (negInstance != null && isUniqueInstance(negInstance, checksumMap, checksumCoeffs)) {
	  instances.add(negInstance);
	  numActualNegatives++;
	  numUniqueNegatives++;
	} else {
 	  numMisfires++;
	} 
      }
      break;

    case NEG_MODE_IMPLICIT_NEGATIVES:
      numMisfires = 0; 
      for (int i = 0; i < numNegPairs && numMisfires < 30000; i++) {
	InstancePair pair = createRandomTrainInstancePair(usedPairSet, checksumMap); 	  
	Instance trainInstance = createInstance(pair, attrIdxs, stringMetrics);
	
	if (trainInstance != null && isUniqueInstance(trainInstance, checksumMap, checksumCoeffs)) {

	  // calculate the fraction of common tokens
	  StringBuffer s1 = new StringBuffer();
	  StringBuffer s2 = new StringBuffer(); 
	  for (int j = 0; j < pair.instance1.numAttributes(); j++) {
	    s1.append(pair.instance1.stringValue(j));
	    s1.append(" ");
	    s2.append(pair.instance2.stringValue(j));
	    s2.append(" ");
	  }
	    
	  if (fractionCommonTokens(s1.toString(), s2.toString()) <= m_maxImplicitCommonTokenFraction) {

	    // check if the negative is bogus
	    if (trainInstance.value(trainInstance.numValues()-1) == 0) {
	      System.out.print("False negative!\n\t" + pair.instance1 + "\n\t" + pair.instance2);

	      if (m_useFalseImplicitNegatives) {
		numActualPositives++; 
	      } else { 
		trainInstance.setValue(trainInstance.numValues()-1, 1);
		numActualNegatives++; 
	      }
	      instances.add(trainInstance);
	    } else { // true implicit negative
	      numActualNegatives++; 
	      instances.add(trainInstance);
	    } 
	  } else {  // try an extra pair if this one didn't work out due to too many positive pairs
	    numMisfires++;
	    i--;
	  }
	} else {
	  // try an extra pair if this one was null or not unique
	  numMisfires++;
	  i--;
	}  
      }
      break;
    }

    System.out.println();
    System.out.println("POSITIVES:  requested=" + numPosPairs + "\tpossible=" + m_numPotentialPositives +
		       "\tactual=" + numActualPositives);
    System.out.println("NEGATIVES:  requested=" + numNegPairs + "\tpossible=" + m_numPotentialNegatives +
		       "\tactual=" + numActualNegatives);
    return instances;
  }

  protected InstancePair createRandomTrainInstancePair(HashSet usedPairSet, HashMap checksumMap) {
    int numTrainingRecords = m_instances.numInstances();
    int idx1, idx2;
    Integer pairCode, pairCodeOrdered;
    int numTries = 0;
    int maxNumTries = 1000;
    InstancePair pair = null;
    Random r = new Random(usedPairSet.size() + checksumMap.size());

    // select a random pair of instances that has not been
    // seen before or until we exhaust all possible pairs
    do {
      idx1 = r.nextInt(numTrainingRecords);
      idx2 = r.nextInt(numTrainingRecords);
      while (idx2 == idx1) { // prevent selecting the same instance twice
	idx2 = r.nextInt(numTrainingRecords);
      }
      pairCode = new Integer(idx1 * numTrainingRecords + idx2);
      pairCodeOrdered = new Integer(idx2 * numTrainingRecords + idx1);
      numTries++;
    } while ((usedPairSet.contains(pairCode) || usedPairSet.contains(pairCodeOrdered))
	     && numTries < maxNumTries);

    if (numTries < maxNumTries) { 
      // create the training instance and add it
      usedPairSet.add(pairCode);
      usedPairSet.add(pairCodeOrdered);
      Instance instance1 = m_instances.instance(idx1);
      Instance instance2 = m_instances.instance(idx2);
      boolean positive = (instance1.classValue() == instance2.classValue());;
      pair = new InstancePair(instance1, instance2, positive, 0);
    }

    return pair; 	  
  } 



  /**
   * Create a nonsparse instance with features corresponding to the
   * metric values between used fields of the two given instances
   * @param instancePair a pair of instances that is used for creating the new diffInstance
   * @param attrIdxs indeces of fields that should be utilized
   * @param metrics the string metrics that are used to create the training instances
   * @return a newly created diffInstance, or null if all diff-values are 0
   */
  protected Instance createInstance (InstancePair pair, int[] attrIdxs, StringMetric[][] metrics ) throws Exception {
    int numAttributes = metrics.length * metrics[0].length + 1;
    int numNonNegativeValues = 0;
    int numValues = 0;
    double[] values = new double[numAttributes];  

    for (int i = 0; i < attrIdxs.length; i++) {
      String val1 = pair.instance1.stringValue(attrIdxs[i]);
      String val2 = pair.instance2.stringValue(attrIdxs[i]);

      for (int j = 0; j < metrics.length; j++) { 
	if (metrics[j][i].isDistanceBased()) { 
	  values[numValues] = metrics[j][i].distance(val1, val2);
	} else {
	  values[numValues] = metrics[j][i].similarity(val1, val2);
	}
	if (values[numValues] != 0) {
	  numNonNegativeValues++;
	}
	numValues++;
      }
    }
    
    if (pair.positive) {
      values[numAttributes-1] = 0;
    } else {
      values[numAttributes-1] = 1;
    }

    // if there were non-zero attributes, return the instance, otherwise return null
    if (numNonNegativeValues > 0) { 
      return new Instance(1.0, values);
    } else {
      return null;
    }
  }



  /** Check whether an instance is unique
   * @param instance instance to be checked
   * @param checksumMap a map where checksum values are mapped to lists of instances
   * @param sumCoeffs coefficients used for computing the checksum
   * @return true if the instance is unique, false otherwise
   */
  protected boolean isUniqueInstance(Instance instance, HashMap checksumMap, double[] checksumCoeffs) {
    double checksum = 0;

    // compute the checksum and round off to overcome machine precision errors
    for (int i = 0; i < instance.numValues()-1; i++) {
      checksum += checksumCoeffs[i] * instance.value(i);
    }
    checksum = (float) checksum;
    
    // if this checksum was encountered before, get a list of instances
    // that have this checksum, and check if any of them are dupes of this one
    if (checksumMap.containsKey(new Double(checksum))) {
      ArrayList checksumList = (ArrayList) checksumMap.get(new Double(checksum));
      boolean unique = true;
      for (int k = 0; k < checksumList.size() && unique; k++) {
	Instance nextDiffInstance = (Instance) checksumList.get(k);
	unique = false;
	for (int l = 0; l < nextDiffInstance.numValues()-1 && !unique; l++) {
	  if (((float)nextDiffInstance.value(l)) != ((float)instance.value(l))) {
	    unique = true;
	  } 
	}
	if (unique == false) {
	  return false;
	}
      }
      checksumList.add(instance);
      return true;  // no dupes were found among instances with the same checksum
    } else {  // this checksum has not been encountered before
      ArrayList checksumList = new ArrayList();
      checksumMap.put(new Double(checksum), checksumList);
      checksumList.add(instance);
      return true;
    }
  }
  
  
  /**
   * Provide an array of string pairs metric using given training instances
   *
   * @param metric the metric to train
   * @param instances data to train the metric on
   * @exception Exception if training has gone bad.
   * @return a list of StringPair's that is training data for a particular field
   */
  public ArrayList getStringPairList(Instances instances, int attrIdx,
				     int numPosPairs, int numNegPairs,
				     StringMetric metric) throws Exception {
    System.out.println("Selecting strings out of " + instances.numInstances() + " instances, first is \n" + instances.instance(0));
    ArrayList pairList = new ArrayList();
    TreeSet posPairSet = null;
    TreeSet negPairSet = null;
    double [] posPairDistances = null;
    double [] negPairDistances = null;
    Iterator iterator = null;
    int numPossiblePosStrPairs = 0, numPossibleNegStrPairs = 0;
    int numActualPositives = 0, numActualNegatives = 0;

    // SELECT POSITIVE PAIRS
    switch (m_posStringMode) {
    case STRING_PAIRS_EASIEST:
      posPairSet = new TreeSet(new ReverseComparator());
      posPairDistances = populatePosStrPairSet(metric, posPairSet, attrIdx);
      numPossiblePosStrPairs = posPairSet.size();
      
      // select numPositives 
      iterator = posPairSet.iterator();
      for (int i = 0; iterator.hasNext() && i < m_numPotentialPositives && i < numPosPairs; i++) {
	StringPair posPair = (StringPair) iterator.next();
	pairList.add(posPair);
      }
      break;

    case STRING_PAIRS_HARDEST:
      posPairSet = new TreeSet();
      posPairDistances = populatePosStrPairSet(metric, posPairSet, attrIdx);
      numPossiblePosStrPairs = posPairSet.size();

      // select numPositives examples
      iterator = posPairSet.iterator();
      for (int i = 0; iterator.hasNext() && i < m_numPotentialPositives && i < numPosPairs; i++) {
	StringPair posPair = (StringPair) iterator.next();
	pairList.add(posPair);
      }
      break;

    case STRING_PAIRS_RANDOM:
      // Get string pairs for a given attribute
      ArrayList strPairList = new ArrayList();
      for (int i = 0; i < m_posPairList.size(); i++) {
	InstancePair pair = (InstancePair) m_posPairList.get(i);
	String str1 = pair.instance1.stringValue(attrIdx);
	String str2 = pair.instance2.stringValue(attrIdx);
	if (!str1.equals(str2) && haveCommonTokens(str1, str2)) {
	  StringPair strPair = new StringPair(str1, str2, true, 0);
	  strPairList.add(strPair);
	} else {
	  System.out.println("Equal strings, or no common tokens - NOT adding: " + str1 + "\t" + str2);
	}
      } 
      numPossiblePosStrPairs = strPairList.size();
      
      // if we have fewer pairs available than requested, return all the ones that were created
      if (strPairList.size() <= numPosPairs) {
	System.out.println("INSUFFICIENT available POSITIVE examples, using all " + strPairList.size());
	pairList = strPairList;
      } else {
	// if we have more than enough potential pairs, sample randomly with replacement
	int[] indexes = randomSubset(numPosPairs, strPairList.size());
	System.out.println("SUFFICIENT available POSITIVE examples, randomly selected " + indexes.length + " of " + strPairList.size());
	for (int i = 0; i < indexes.length; i++) {
	  pairList.add(strPairList.get(indexes[i]));
	}
      }
      //  	for (int i =0 ; i < strPairList.size(); i++) {
      //  	  StringPair pair = (StringPair) strPairList.get(i);
      //  	  System.out.println(pair.str1 + "\t\t\t" + pair.str2);
      //  	}
      
      break;

    default:
      throw new Exception("Unknown method for selecting positive pairs: " + m_posStringMode);
    }
    numActualPositives = pairList.size();

    // we don't need negative string pairs for AffineProbMetric
    if (!metric.getClass().getName().equals("weka.deduping.metrics.AffineProbMetric")) {
      
      // SELECT NEGATIVE PAIRS unless this is AffineProbMetric - it doesn't need negatives
      switch (m_negStringMode) {
      case STRING_PAIRS_EASIEST:
	// Create a map with *all* negatives
	negPairSet = new TreeSet();
	negPairDistances = populateNegStrPairSet(metric, negPairSet, attrIdx);
	numPossibleNegStrPairs = negPairSet.size();
      
	iterator = negPairSet.iterator();
	for (int i = 0; iterator.hasNext() && i < m_numPotentialNegatives && i < numNegPairs; i++) {
	  StringPair negPair = (StringPair) iterator.next();
	  pairList.add(negPair);
	  System.out.println("EASY:   " + negPair.value + "\n\t" + negPair.str1 + "\n\t" + negPair.str2);
	}
	break;

      case STRING_PAIRS_HARDEST:
	negPairSet = new TreeSet(new ReverseComparator());
	negPairDistances = populateNegStrPairSet(metric, negPairSet, attrIdx);
	numPossibleNegStrPairs = negPairSet.size();

	// We will hash each pair of classes that was used so that we don't end up with
	// too many pairs from the same combination of two classes
	HashSet usedComboSet = new HashSet();
          
	iterator = negPairSet.iterator();
	for (int i = 0; iterator.hasNext() && i < m_numPotentialNegatives && i < numNegPairs; i++) {
	  StringPair negPair = (StringPair) iterator.next();
	  Double class1class2HashValue = new Double(negPair.class1 * 100000 + negPair.class2);
	  if (!usedComboSet.contains(class1class2HashValue)) { //  kludge - comment out for cora1
	    pairList.add(negPair);
	    //	  System.out.println("HARD:   " + negPair.value + "\n\t" + negPair.str1 + "\n\t" + negPair.str2);
	    usedComboSet.add(class1class2HashValue);

	    // add reverse combo (or allow two per class if commented out
	    //	  usedComboSet.add(new Double(negPair.class2 * 1000 + negPair.class1));  <- reverse combo
	  } 	  
	}
	break;

      case STRING_PAIRS_RANDOM:
	// Get string pairs for a given attribute
	ArrayList strPairList = new ArrayList();
	for (int i = 0; i < m_negPairList.size(); i++) {
	  InstancePair pair = (InstancePair) m_negPairList.get(i);
	  String str1 = pair.instance1.stringValue(attrIdx);
	  String str2 = pair.instance2.stringValue(attrIdx);
	  if (!str1.equals(str2)) {
	    StringPair strPair = new StringPair(str1, str2, false, 0);
	    strPairList.add(strPair);
	  }
	}
	numPossibleNegStrPairs = strPairList.size();
      
      
	// if we have fewer pairs available than requested, return all the ones that were created
	if (strPairList.size() <= numNegPairs) {
	  System.out.println("INSUFFICIENT available NEGATIVE examples, using all " + strPairList.size());
	  pairList.addAll(strPairList);
	} else { // if we have enough potential pairs, randomly sample with replacement
	  int[] indexes = randomSubset(numNegPairs, strPairList.size());
	  System.out.println("SUFFICIENT available NEGATIVE examples, randomly selected " + indexes.length + " of " + strPairList.size());
	  for (int i = 0; i < indexes.length; i++) {
	    pairList.add(strPairList.get(indexes[i]));
	  }
	}
	break;
	
      default:
	throw new Exception("Unknown method for selecting negative pairs: " + m_negStringMode);
      }
    }
    numActualNegatives = pairList.size() - numActualPositives;

    System.out.println();
    System.out.println("**POSITIVES:  requested=" + numPosPairs + "\tpossible=" + numPossiblePosStrPairs +
		       "\tactual=" + numActualPositives);
    System.out.println("**NEGATIVES:  requested=" + numNegPairs + "\tpossible=" + numPossibleNegStrPairs +
		       "\tactual=" + numActualNegatives);
    return pairList;
  }

  /** Add a pair to a TreeSet so that there are no collisions, and no values are erased
   * @param set a set to which a new pair should be added
   * @param pair a new pair of strings that is to be added; value
   * fields holds the distance between the strings
   * @return the unique value of the distance (possibly perturbed)
   * with which the pair was added
   */
  protected double addUniquePair(TreeSet set, StringPair pair) {
    Random random = new Random();
    double epsilon = 0.0000001;
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
      if (counter % 10 == 0) {  // increase perturbations if "nearby" values have been exhausted
	epsilon *= 10;
      }
    }
    set.add(pair);
    return pair.value;
  }

  
  /** Populate a provided treeset with all positive StringPair's
   * @param metric a metric that will be used to calculate distance
   * @param pairSet an empty TreeSet that will be populated
   * @param attrIdx the index of the attribute for which positive
   * string pairs are being accumulated
   * @return an array with distance values of the created pairs
   */
  protected double[] populatePosStrPairSet(StringMetric metric, TreeSet strPairSet, int attrIdx) throws Exception {
    double [] posPairDistances = new double[m_numPotentialPositives];
    Arrays.fill(posPairDistances, Double.MIN_VALUE);
    int posCounter = 0;

    for (int i = 0; i < m_posPairList.size(); i++) {
      InstancePair pair = (InstancePair) m_posPairList.get(i);
      String str1 = pair.instance1.stringValue(attrIdx);
      String str2 = pair.instance2.stringValue(attrIdx);
      // unless the two fields are exact duplicates, create a new pair
      if (!str1.equals(str2)) {
	StringPair strPair = new StringPair(str1, str2, true, metric.similarity(str1, str2));
	posPairDistances[posCounter++] = addUniquePair(strPairSet, strPair);
      }      
    } 
    return posPairDistances;
  }

  /** Populate a provided treeset with a sufficient population of negative StringPair's
   * @param metric a metric that will be used to calculate distance between strings
   * @param pairSet an empty TreeSet that will be populated
   * @param attrIdx the index of the attribute for which positive
   * string pairs are being accumulated
   * @return an array with distance values of the created pairs
   */
  protected double[] populateNegStrPairSet(StringMetric metric, TreeSet strPairSet, int attrIdx) throws Exception {
    // Create a map with *all* positives
    double [] negPairDistances = new double[m_numPotentialNegatives];
    Arrays.fill(negPairDistances, Double.MIN_VALUE);
    int negCounter = 0;
    int[] negPairIdxs;

    // get a random sample if we have too many possible negatives  TODO - are we limiting ourselves here???
    negPairIdxs = randomSubset(20000, m_numPotentialNegatives);

    for (int i = 0; i < negPairIdxs.length; i++) {
      InstancePair pair = (InstancePair) m_negPairList.get(negPairIdxs[i]);
      String str1 = pair.instance1.stringValue(attrIdx);
      String str2 = pair.instance2.stringValue(attrIdx);
      // unless the two fields are exact duplicates, create a new pair
      if (!str1.equals(str2)) {
	StringPair strPair = new StringPair(str1, str2, false, metric.similarity(str1, str2));
	strPair.class1 = pair.instance1.classValue();
	strPair.class2 = pair.instance2.classValue();
	negPairDistances[negCounter++] = addUniquePair(strPairSet, strPair);
      }      
    } 
    
    return negPairDistances;
  }
  
  
  /** Populate m_posPairList with all positive InstancePair's */
  protected void createPosPairList() {
    // go through lists of instances for each class and create a list of *all* positive pairs
    m_posPairList = new ArrayList();
    Iterator iterator = m_classInstanceMap.values().iterator();
    while (iterator.hasNext()) {
      ArrayList instanceList = (ArrayList) iterator.next();
      // create all same-class pairs for every true object
      for (int i = 0; i < instanceList.size(); i++) {
	Instance instance1 = (Instance) instanceList.get(i);
	for (int j = i+1; j < instanceList.size(); j++) {
	  Instance instance2 = (Instance) instanceList.get(j);
	  InstancePair pair = new InstancePair(instance1, instance2, true, 0);
	  m_posPairList.add(pair);
	}
      }
    }
  }

  /** Populate m_negPairList with negative InstancePair's */
  protected void createNegPairList() {
    m_negPairList = new ArrayList();
    // go through lists of instances for each class
    for (int i = 0; i < m_classValueList.size(); i++) {
      ArrayList instanceList1 = (ArrayList) m_classInstanceMap.get(m_classValueList.get(i));
      for (int j = 0; j < instanceList1.size(); j++) {
	Instance instance1 = (Instance) instanceList1.get(j);
	// create all pairs from other clusters with this str
	for (int k = i+1; k < m_classValueList.size(); k++) {
	  ArrayList instanceList2 = (ArrayList) m_classInstanceMap.get(m_classValueList.get(k));
	  for (int l = 0; l < instanceList2.size(); l++) {
	    Instance instance2 = (Instance) instanceList2.get(l);
	    InstancePair pair = new InstancePair(instance1, instance2, false, 0);
	    m_negPairList.add(pair);
	  }
	}
      }
    }
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
   * @param mode selection mode for positive examples
   */
  public void setPositivesMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_POS_MODE) {
      m_positivesMode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the selection mode for positives
   * @return one of the selection modes
   */
  public SelectedTag getPositivesMode() {
    return new SelectedTag(m_positivesMode, TAGS_POS_MODE);
  }


  /** Set the selection mode for negatives
   * @param mode selection mode for negative examples
   */
  public void setNegativesMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_NEG_MODE) {
      m_negativesMode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the selection mode for negatives
   * @return one of the selection modes
   */
  public SelectedTag getNegativesMode() {
    return new SelectedTag(m_negativesMode, TAGS_NEG_MODE);
  }

  /**
   * Set the maximum fraction of common tokens that instances can have to
   * be included as implicit negatives
   * @param maxImplicitCommonTokenFraction
   */
  public void setMaxImplicitCommonTokenFraction(double maxImplicitCommonTokenFraction) {
    m_maxImplicitCommonTokenFraction = maxImplicitCommonTokenFraction;
  } 

  /**
   * Get the maximum fraction of common tokens that instances can have to
   * be included as implicit negatives
   * @return the fraction
   */
  public double getMaxImplicitCommonTokenFraction() {
    return m_maxImplicitCommonTokenFraction;
  }

  
  /** Turn using rejected positives as negatives on/off
   * @param useRejectedPositives if true, false positives that were picked during the
   * static-active selection will be added to the negatives set
   */
  public void setUseRejectedPositives(boolean useRejectedPositives) {
    m_useRejectedPositives = useRejectedPositives;
  }

  /** Check whether using rejected positives as negatives is on or off
   * @return returns true if false positives that were picked during
   * the static-active selection are added to the negatives set
   */
  public boolean getUseRejectedPositives() {
    return m_useRejectedPositives;
  }


  /** Turn using false implicit negatives on/off
   * @param useFalseImplicitNegatives if true, false implicit negatives will be added to  positives
   */
  public void setUseFalseImplicitNegatives(boolean useFalseImplicitNegatives) {
    m_useFalseImplicitNegatives = useFalseImplicitNegatives;
  }

  /** Check whether using false implicit negatives is on/off
   * @return true if false implicit negatives are added to  positives
   */
  public boolean getUseFalseImplicitNegatives() {
    return m_useFalseImplicitNegatives;
  }

  
  /** Set the selection mode for positive string examples
   * @param mode selection mode for positive  string examples
   */
  public void setPosStringMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_STRING_PAIR_MODE) {
      m_posStringMode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the selection mode for positive string examples
   * @return one of the selection modes for positive  string examples
   */
  public SelectedTag getPosStringMode() {
    return new SelectedTag(m_posStringMode, TAGS_STRING_PAIR_MODE);
  }


  
  /** Set the selection mode for negative string examples
   * @param mode selection mode for negative  string examples
   */
  public void setNegStringMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_STRING_PAIR_MODE) {
      m_negStringMode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the selection mode for negative string examples
   * @return one of the selection modes for negative  string examples
   */
  public SelectedTag getNegStringMode() {
    return new SelectedTag(m_negStringMode, TAGS_STRING_PAIR_MODE);
  }


  
  
  /** Turn debugging output on/off
   * @param debug if true, debugging info will be printed
   */
  public void setDebug(boolean debug) {
    m_debug = debug;
  }

  /** See whether debugging output is on/off
   * @returns if true, debugging info will be printed
   */
  public boolean getDebug() {
    return m_debug;
  }

  
  /**
   * get an array random indeces out of n possible values.
   * if the number of requested indeces is larger then maxIdx, returns
   * maxIdx permuted values
   * @param maxIdx - the maximum index of the set
   * @param numIdxs number of indexes to return
   * @return an array of indexes
   */
  public static int[] randomSubset(int numIdxs, int maxIdx) {
    Random r = new Random(maxIdx + numIdxs);
    int[] indeces = new int[maxIdx];

    for (int i = 0; i < maxIdx; i++) {
      indeces[i] = i;
    }

    // permute the indeces randomly
    for (int i = 0; i < indeces.length; i++) {
      int idx = r.nextInt (maxIdx);
      int temp = indeces[idx];
      indeces[idx] = indeces[i];
      indeces[i] = temp;
    }
    int []returnIdxs = new int[Math.min(numIdxs,maxIdx)];
    for (int i = 0; i < returnIdxs.length; i++) {
      returnIdxs[i] = indeces[i];
    }
    return returnIdxs;
  }


  /** return true if two strings have commmon tokens */
  public static boolean haveCommonTokens(String s1, String s2) {
    String delimiters = " \t\n\r\f\'\"\\!@#$%^&*()_-+={}<>,.;:|[]{}/*~`";
    
    HashSet tokenSet1 = new HashSet(); 
    StringTokenizer tokenizer = new StringTokenizer(s1, delimiters);
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      tokenSet1.add(token);
    }

    int count = 0; 
    tokenizer = new StringTokenizer(s2, delimiters);
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      if (tokenSet1.contains(token)) {
	count++;
	if (count > 0) {
	  return true;
	}
      }
    }
    return false;
  } 

  /** return the number of commmon tokens that two strings have
   * @param s1 string 1
   * @param s2 string 2
   * @return the number of common tokens the strings have
   */
  public static double fractionCommonTokens(String s1, String s2) {
    String delimiters = " \t\n\r\f\'\"\\!@#$%^&*()_-+={}<>,.;:|[]{}/*~`";
    
    HashSet tokenSet1 = new HashSet();
    int commonTokens = 0;
    int totalTokens = 0;
    
    StringTokenizer tokenizer = new StringTokenizer(s1, delimiters);
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      tokenSet1.add(token);
      totalTokens++;
    }

    tokenizer = new StringTokenizer(s2, delimiters);
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      if (tokenSet1.contains(token)) {
	commonTokens++;
      }
      totalTokens++;
    }
    return ((commonTokens + 0.0)/totalTokens);
  } 
  

  

  /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [10];
    int current = 0;

    switch(m_positivesMode) {
    case POS_MODE_RANDOM_RECORDS:
      options[current++] = "-Pr";
      break;
    case POS_MODE_RANDOM_POSITIVES:
      options[current++] = "-Pp";
      break;
    case POS_MODE_STATIC_ACTIVE:
      if (m_useRejectedPositives) { 
	options[current++] = "-PsN";
      } else {
	options[current++] = "-Ps";
      }
      break;
    }

    switch(m_negativesMode) {
    case NEG_MODE_RANDOM_RECORDS:
      options[current++] = "-Nr";
      break;
    case NEG_MODE_RANDOM_NEGATIVES:
      options[current++] = "-Nn";
      break;
    case NEG_MODE_IMPLICIT_NEGATIVES:
      if (m_useFalseImplicitNegatives) { 
	options[current++] = "-NiP" + m_maxImplicitCommonTokenFraction;
      } else {
	options[current++] = "-Ni" + m_maxImplicitCommonTokenFraction;
      }
      break;
    }
    

    switch(m_posStringMode) {
    case STRING_PAIRS_RANDOM:
      options[current++] = "-SPr";
      break;
    case STRING_PAIRS_HARDEST:
      options[current++] = "-SPh";
      break;
    case STRING_PAIRS_EASIEST:
      options[current++] = "-SPe";
      break;
    }

    switch(m_negStringMode) {
    case STRING_PAIRS_RANDOM:
      options[current++] = "-SNr";
      break;
    case STRING_PAIRS_HARDEST:
      options[current++] = "-SNh";
      break;
    case STRING_PAIRS_EASIEST:
      options[current++] = "-SNe";
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
}
