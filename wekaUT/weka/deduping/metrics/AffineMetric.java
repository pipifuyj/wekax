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
 *    AffineDistance.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */

package weka.deduping.metrics;

import java.util.*;
import java.io.Serializable;
import weka.core.*;



/**
 * A measure of distance between two strings based on affine distance.
 * See D. Gusfield, "Algorithms on Strings, Trees and Sequences",
 * Cambridge University Press, 1997. 
 *
 * @author Mikhail Bilenko
 */

public class AffineMetric extends StringMetric implements OptionHandler, Serializable {

  /** The cost of matching two characters */
  protected double m_matchCost = -1;
  
  /** The cost of a substituting two characters */
  protected double m_subCost = 2;
  
  /** The cost of opening a gap */
  protected double m_gapStartCost = 3;
  
  /** The cost of continuing a gap */
  protected double m_gapExtendCost = 1;

  /** Should the distance be normalized by the lengths of the strings? */
  protected boolean m_normalized = true;

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
  protected int m_conversionType = CONVERSION_EXPONENTIAL;

    
  /** A default constructor that assigns the name of this distance */
  public AffineMetric () { }

  /** A metric can be data-dependent (e.g. vector space for IDF) */
  public boolean isDataDependent() {
    return false;
  }


  /** Obtain the distance between two strings
   * @param s1 String 1
   * @param s2 String 2
   * @returns Affine distance between the two strings 
   */
  public  double distance(String string1, String string2) throws Exception {
    char[] s1 = string1.toCharArray();
    char[] s2 = string2.toCharArray();
    int l1 = s1.length, l2 = s2.length;
    double T[][] = new double[l1+1][l2+1];
    double I[][] = new double[l1+1][l2+1];
    double D[][] = new double[l1+1][l2+1];
    double subCost;
    int i, j;

    if (l1==0 || l2==0) {
      return m_gapStartCost + (l1+l2-1) * m_gapExtendCost;
    }
    for (j = 0; j < l2+1; j++) {
      I[0][j] = Double.MAX_VALUE;
      D[0][j] = Double.MAX_VALUE;
    }
    for (j = 0; j < l1+1; j++) {
      I[j][0] = Double.MAX_VALUE;
      D[j][0] = Double.MAX_VALUE;
    }
    T[0][0] = 0;
    T[0][1] = m_gapStartCost;
    T[1][0] = m_gapStartCost;
    for (j = 2; j < l2+1; j++) {
      T[0][j] = T[0][j-1] + m_gapExtendCost;
    }
    for (j = 2; j < l1+1; j++) {
      T[j][0] = T[j-1][0] + m_gapExtendCost;
    }
    for (i = 1; i < l1+1; i++) {
      for (j = 1; j < l2+1; j++) {
	D[i][j] = (D[i-1][j]+m_gapExtendCost > T[i-1][j]+m_gapStartCost) ?
	  T[i-1][j]+m_gapStartCost : D[i-1][j]+m_gapExtendCost;
	I[i][j] = (I[i][j-1]+m_gapExtendCost > T[i][j-1]+m_gapStartCost) ?
	  T[i][j-1]+m_gapStartCost : I[i][j-1]+m_gapExtendCost;
	subCost = (s1[i-1] == s2[j-1]) ? m_matchCost : m_subCost;
	if  ((T[i-1][j-1] + subCost < D[i][j]) && (T[i-1][j-1] + subCost < I[i][j])) {
	  T[i][j] = T[i-1][j-1] + subCost;
	} else {
	  if (D[i][j] < I[i][j]) {
	    T[i][j] = D[i][j];
	  } else {
	    T[i][j] = I[i][j];
	  }
	}
      }
    }
    double ret;
    if (T[l1][l2] < D[l1][l2] && T[l1][l2] < I[l1][l2]) {
      ret = T[l1][l2];
    } else if (D[l1][l2] < I[l1][l2]) {
      ret = D[l1][l2];
    } else {
      ret = I[l1][l2];
    }
    if (m_normalized) {
      ret /= l1 + l2;
    }
    return ret;
  }


  /** The computation of a metric can be either based on distance, or on similarity
   * @returns true
   */
  public boolean isDistanceBased() {
    return true;
  }

  /**
   * Returns a similarity estimate between two strings. Similarity is obtained by
   * inverting the distance value using one of three methods:
   * CONVERSION_LAPLACIAN, CONVERSION_EXPONENTIAL, CONVERSION_UNIT.
   * @param string1 First string.
   * @param string2 Second string.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarity(String string1, String string2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + distance(string1, string2));
    case CONVERSION_UNIT:
      return 2 * (1 - distance(string1, string2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-distance(string1, string2));
    default:
      throw new Exception ("Unknown distance to similarity conversion method");
    }
  }

  /** Set the match cost
   * @param matchCost the cost of finding a matching pair of characters
   */
  public void setMatchCost(double matchCost) {
    m_matchCost = matchCost;
  } 

  /** Get the match cost
   * @returns the cost of finding a matching pair of characters
   */
  public double getMatchCost() {
    return m_matchCost;
  } 

  /** Set the substitution cost
   * @param subCost the cost of substituting one character for another
   */
  public void setSubCost(double subCost) {
    m_subCost = subCost;
  } 

  /** Get the substitution cost
   * @returns the cost of substituting a pair of characters
   */
  public double getSubCost() {
    return m_subCost;
  }

  /** Set the gap opening cost
   * @param gapStartCost the cost of opening a gap
   */
  public void setGapStartCost(double gapStartCost) {
    m_gapStartCost = gapStartCost;
  } 

  /** Get the gap opening cost
   * @returns the cost of opening a gap
   */
  public double getGapStartCost() {
    return m_gapStartCost;
  }

  /** Set the gap extension cost
   * @param gapExtendCost the cost of extending a gap
   */
  public void setGapExtendCost(double gapExtendCost) {
    m_gapExtendCost = gapExtendCost;
  } 

  /** Get the gap extension cost
   * @returns the cost of extending a gap
   */
  public double getGapExtendCost() {
    return m_gapExtendCost;
  }


  /** Set the distance to be normalized by the sum of the string's lengths
   * @param normalized if true, distance is normalized by the sum of string's lengths
   */
  public void setNormalized(boolean normalized) {
    m_normalized = normalized;
  } 

  /** Get whether the distance is normalized by the sum of the string's lengths
   * @return if true, distance is normalized by the sum of string's lengths
   */
  public boolean getNormalized() {
    return m_normalized;
  }

  /** Create a copy of this metric
   * @return another AffineMetric with the same exact parameters as this  metric
   */
  public Object clone() {
    AffineMetric metric = new AffineMetric();
    metric.setNormalized(m_normalized);
    metric.setMatchCost(m_matchCost);
    metric.setSubCost(m_subCost); 
    metric.setGapStartCost(m_gapStartCost);
    metric.setGapExtendCost(m_gapExtendCost);
    return metric;
  }

  /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [10];
    int current = 0;

    if (m_normalized) {
      options[current++] = "-N";
    }
    options[current++] = "-m";
    options[current++] = "" + m_matchCost;
    options[current++] = "-s";
    options[current++] = "" + m_subCost;
    options[current++] = "-g";
    options[current++] = "" + m_gapStartCost;
    options[current++] = "-e";
    options[current++] = "" + m_gapExtendCost;
    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -N normalize by length
   * -m matchCost
   * -s subCost
   * -g gapStartCost
   * -e gapExtendCost   
   */
  public void setOptions(String[] options) throws Exception {
    setNormalized(Utils.getFlag('N', options));
    
    String matchCostString = Utils.getOption('m', options);
    if (matchCostString.length() != 0) {
      setMatchCost(Double.parseDouble(matchCostString));
    }

    String subCostString = Utils.getOption('s', options);
    if (subCostString.length() != 0) {
      setSubCost(Double.parseDouble(subCostString));
    }

    String gapStartString = Utils.getOption('g', options);
    if (gapStartString.length() != 0) {
      setGapStartCost(Double.parseDouble(gapStartString));
    }

    String gapExtendString = Utils.getOption('e', options);
    if (gapExtendString.length() != 0) {
      setGapExtendCost(Double.parseDouble(gapExtendString));
    } 
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector(5);

    newVector.addElement(new Option("\tNormalize the dot product by vectors lengths\n",
				    "N", 0, "-N"));
    newVector.addElement(new Option("\tMatch cost\n",
				    "m", 1, "-m matchCost"));
    newVector.addElement(new Option("\tSubstitution cost\n",
				    "s", 1, "-m subCost"));
    newVector.addElement(new Option("\tGap start cost\n",
				    "g", 1, "-g gapStartCost"));
    newVector.addElement(new Option("\tGap extend cost\n",
				    "e", 1, "-e gapExtendCost"));
    
    return newVector.elements();
  }
}











