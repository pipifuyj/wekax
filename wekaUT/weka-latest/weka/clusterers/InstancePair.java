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
 *    InstancePair.java
 *    Copyright (C) 2002 Sugato Basu
 *
 */

package weka.clusterers;

import java.util.*;
import weka.core.Instance;
import weka.core.Instances;

/** Class for handling a pair of instances, in terms of indices of instances in an Instances set */

public class InstancePair implements Comparable {
  /** first instance index */  
  public int first; 
  /** second instance index, always <= first */
  public int second; 
  /** MUST_LINK, CANNOT_LINK or DONT_CARE_LINK */
  public int linkType;
  /** cost of violating constraint */
  public double cost;

  /** score from active learning algorithm */
  /** ----- DEPRECATED: ACTIVE SCORE NO LONGER USED IN PCKMEANS!!!! -----*/
  public double activeScore;

  /** must-link */
  public final static int MUST_LINK = 29;
  /** cannot-link */
  public final static int CANNOT_LINK = 31;
  /** don't care */
  public final static int DONT_CARE_LINK = 37;

  public static boolean m_isClassAttributeString = false;

  /** constructor */
  public InstancePair() {
  }

  /** constructor */
  public InstancePair(int a, int b) {
    first = a;
    second = b;
  }

  /** constructor */
  public InstancePair(int a, int b, int l) {
    first = a;
    second = b;
    linkType = l;
  }

  /** constructor */
  public InstancePair(int a, int b, int l, double c) {
    first = a;
    second = b;
    linkType = l;
    cost = c;
  }

  /** Compare function
   * @return 0 if equal, -1 if this.activeScore > a.activeScore, +1 else
   * Note: Reverse of conventional compareTo, to force sort in descending order
   */
  public int compareTo (Object a) {
    if (a instanceof InstancePair) {
      return compareTo((InstancePair)a);
    }
    return 0;
  }

  /** Compare function
   * @return 0 if equal, -1 if this.activeScore > a.activeScore, +1 else
   * Note: Reverse of conventional compareTo, to force sort in descending order
   */
  public int compareTo (InstancePair a) {
    if (this.activeScore == a.activeScore) 
      return 0;
    else if (this.activeScore > a.activeScore) 
      return -1;
    return +1;
  }


  /** Equals function
   * @return true if same, false else
   */
  public boolean equals (Object a) {
    if (a instanceof InstancePair) {
      InstancePair b = (InstancePair) a;
      if (this.first==b.first &&
	  this.second==b.second && 
	  this.linkType==b.linkType) {
	return true;
      }
      else {
	return false;
      }
    }
    return super.equals(a);
  }

  /** hashCode */
  public int hashCode() {
    return first*second*linkType;
  }

  /** Finds whether index is in pair */
  boolean contains (int num) {
    return (first == num || second == num);
  }

  /** Returns an arraylist of random (both positive and negative) pair objects created from the input
   *  @param instances list of instances
   *  @param size number of pairs to return
   *  @return arraylist of pairs
   */
  public static ArrayList getPairs(Instances instances, int size) {
    return getPairs(instances, size, -1);
  }


  /** Returns an arraylist of pair objects created from the input set of instances
   *  @param instances list of instances
   *  @param size number of pairs to return
   *  @param fractionMustLinks proportion of Must-Links; if -1 - sample randomly
   *  @return arraylist of pairs
   */
  public static ArrayList getPairs(Instances instances, int size, double fractionMustLinks) {
    ArrayList pairs = new ArrayList(size);
    int num=0;
    Random rand = new Random(42);

    m_isClassAttributeString = instances.instance(0).classAttribute().isString();

    if (fractionMustLinks != -1) {
      int numMustLinks = (int) (fractionMustLinks * size);
      int numCannotLinks = size - numMustLinks;
      int numClasses = instances.numClasses();
	  
      // stratify instances into lists for each class
      HashMap classListMap = new HashMap();
      for (int i = 0; i < instances.numInstances(); i++) {
	Double classValue = new Double(instances.instance(i).classValue());
	if (classListMap.containsKey(classValue)) {
	  ArrayList classList = (ArrayList) classListMap.get(classValue);
	  classList.add(new Integer(i)); 
	} else { // previously unseen class
	  ArrayList classList = new ArrayList();
	  classList.add(new Integer(i)); 
	  classListMap.put(classValue, classList); 
	} 
      }

      // select must-links first
      while (num < numMustLinks) {
	int first = rand.nextInt(instances.numInstances());
	int second = 0;

	if (!m_isClassAttributeString) {
	  Double classValue = new Double(instances.instance(first).classValue());
	  ArrayList classList = (ArrayList) classListMap.get(classValue);
	  // skip classes with a single instance
	  if (classList.size() < 2) {
	    continue;
	  }
	  // select a random instance from the same class
	  int idx = rand.nextInt(classList.size());
	  second = ((Integer) classList.get(idx)).intValue();
	} else { // phylo profile case
	  second = rand.nextInt(instances.numInstances());
	  while (second == first) {
	    second = rand.nextInt(instances.numInstances());
	  }
	}
	if (first > second) { // flip if out of order
	  int i = first;
	  first = second;
	  second = i;
	}

	Instance firstInstance = instances.instance(first);
	Instance secondInstance = instances.instance(second);

	if (m_isClassAttributeString) {
	  // for handling string valued class attributes corr. to
	  // multi-class phylogenetic profiles
	  double jaccardSim = jaccardSimilarityOfClassStrings(firstInstance, secondInstance);
	  int linkType = InstancePair.DONT_CARE_LINK; 
	  double cost = 0;
	  if (jaccardSim > 0) {	    
	    linkType = InstancePair.MUST_LINK;
	    cost = jaccardSim;
	  } else if (jaccardSim == 0) {
	    linkType = InstancePair.CANNOT_LINK;
	    cost = 1.0;
	  } else { // jaccardSim < 0 => don't care link
	    linkType = InstancePair.DONT_CARE_LINK;
	    cost = -1.0;
	  }
	  InstancePair pair = new InstancePair(first, second, linkType, cost);
	  if (first!=second && !pairs.contains(pair) && linkType == InstancePair.MUST_LINK && cost < 1.0) { // to filter homologs
	    pairs.add(pair);
	    //  	    System.out.println("Instances are:\n" + firstInstance + "\n" + secondInstance);
	    //  	    System.out.println("Jaccard sim = " + cost);
	    //  	    System.out.println(num + "th pair is: " + pair);
	    num++;
	  }	  
	} else {
	  int linkType = (instances.instance(first).classValue() == 
			  instances.instance(second).classValue())? 
	    InstancePair.MUST_LINK:InstancePair.CANNOT_LINK;
	  InstancePair pair = new InstancePair(first, second, linkType);
	  if (first != second && !pairs.contains(pair) && linkType == InstancePair.MUST_LINK) {
	    pairs.add(pair);
	    num++;
	  }
	}
      }

      // now add cannot-links - NB:  for now not dealing with string attributes; TODO: handle m_isClassAttributeString
      num = 0;
      while (num < numCannotLinks) {
	// we just sample randomly - arguably less time-efficient, but we don't need to
	// create another hash this way.
	int first = rand.nextInt(instances.numInstances());
	int second = rand.nextInt(instances.numInstances());
	while (instances.instance(first).classValue() == instances.instance(second).classValue()) {
	  second = rand.nextInt(instances.numInstances());
	}
	if (first > second) { // flip if out of order
	  int i = first;
	  first = second;
	  second = i;
	}
	InstancePair pair = new InstancePair(first, second, InstancePair.CANNOT_LINK);
	if (!pairs.contains(pair)) {
	  pairs.add(pair);
	  num++;
	}
      }
      System.out.println("Created " + numMustLinks + " must-links and " + numCannotLinks + " cannot-links."); 
    } else { // just collect the requested number of instance pairs by sampling randomly
      while (num < size) {
	int i = rand.nextInt(instances.numInstances());
	int j = rand.nextInt(instances.numInstances());
	int first = (i<j)? i:j;
	int second = (i>=j)? i:j;
	Instance firstInstance = instances.instance(first);
	Instance secondInstance = instances.instance(second);
	if (firstInstance.classAttribute().isString()) {
	  // for handling string valued class attributes corr. to
	  // multi-class phylogenetic profiles
	  double jaccardSim = jaccardSimilarityOfClassStrings(firstInstance, secondInstance);
	  int linkType = InstancePair.DONT_CARE_LINK; 
	  double cost = 0;
	  if (jaccardSim > 0) {	    
	    linkType = InstancePair.MUST_LINK;
	    cost = jaccardSim;
	  } else if (jaccardSim == 0) {
	    linkType = InstancePair.CANNOT_LINK;
	    cost = 1.0;
	  } else { // jaccardSim < 0 => don't care link
	    linkType = InstancePair.DONT_CARE_LINK;
	    cost = -1.0;
	  }
	  InstancePair pair = new InstancePair(first, second, linkType, cost);
	  if (first!=second && !pairs.contains(pair) && linkType != InstancePair.DONT_CARE_LINK) {
	    pairs.add(pair);
	    //	    System.out.println(num + "th pair is: " + pair);
	    num++;
	  }	  
	} else {
	  int linkType = (instances.instance(first).classValue() == 
			  instances.instance(second).classValue())? 
	    InstancePair.MUST_LINK:InstancePair.CANNOT_LINK;
	  InstancePair pair = new InstancePair(first, second, linkType);
	  if (first!=second && !pairs.contains(pair)) {
	    pairs.add(pair);
	    //	System.out.println(num + "th pair is: " + pair);
	    num++;
	  }
	}
      }
    }
    return pairs;
  }

  public static double jaccardSimilarityOfClassStrings(Instance a, Instance b) {
    String s1 = a.classAttribute().value((int) a.classValue());
    String s2 = b.classAttribute().value((int) b.classValue());

    //    System.out.println("Trying out " + s1 + " and " + s2);
    int numTokens1 = 0, numTokens2 = 0, numCommonTokens = 0;
    HashSet set1 = new HashSet();
    StringTokenizer tokenizer = new StringTokenizer(s1, "_");
    while (tokenizer.hasMoreTokens()) {
      set1.add(tokenizer.nextToken());
      numTokens1++;
    }
    
    tokenizer = new StringTokenizer(s2, "_");
    while (tokenizer.hasMoreTokens()) {
      if (set1.contains(tokenizer.nextToken())) {
	numCommonTokens++;
      }
      numTokens2++;
    }
  
    double jaccSim = 0;
    if (numTokens1 + numTokens2 > 0) {
      jaccSim = (numCommonTokens + 0.0) / (numTokens1 + numTokens2 - numCommonTokens);
    }
    if (numTokens1 == 0 || numTokens2 == 0) {
      jaccSim = -1; // to indicate DONT_CARE_LINK
    }

    //      System.out.println("Instances are:\n" + a + "\n" + b);
    //      System.out.println("Jaccard sim of " + s1 + " and " + s2 + " = " + jaccSim);
    return jaccSim;
  }

  /** returns string representation of InstancePair 
   */

  public String toString() {
    String string = new String();
    string = "[" + first + "," + second + ",";
    if (linkType == MUST_LINK) {
      string = string + "MUST,";
    }
    else if (linkType == CANNOT_LINK) {
      string = string + "CANNOT,";
    }
    else if (linkType == DONT_CARE_LINK) {
      string = string + "DONTCARE,";
    }
    string += cost + "]";
    return string;
  }
}

