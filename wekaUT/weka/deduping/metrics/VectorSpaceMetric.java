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
 *    VectorSpaceMetric.java
 *    Copyright (C) 2001 Mikhail Bilenko, Raymond J. Mooney
 *
 */


package weka.deduping.metrics;

import java.util.*;
import java.io.Serializable;
import weka.core.*;

/**
 * This class uses a vector space to calculate similarity between two strings
 * Some code borrowed from ir.vsr package by Raymond J. Mooney
 *
 * @author MikhailBilenko
 */

public class VectorSpaceMetric extends StringMetric implements DataDependentStringMetric,
							       OptionHandler, Serializable {

  /** Strings are mapped to StringReferences in this hash */
  protected HashMap m_stringRefHash = null;

  /** A HashMap where tokens are indexed. Each indexed token maps
   * to a TokenInfo. */
  protected HashMap m_tokenHash = null;

  /** A list of all indexed strings.  Elements are StringReference's. */
  public ArrayList m_stringRefs = null;

  /** An underlying tokenizer that is used for converting strings
   * into HashMapVectors
   */
  protected Tokenizer m_tokenizer = new WordTokenizer();

  /** Should IDF weighting be used? */
  protected boolean m_useIDF = true;

  /** We can have different ways of converting from similarity to distance */
  public static final int CONVERSION_LAPLACIAN = 1;
  public static final int CONVERSION_UNIT = 2;
  public static final int CONVERSION_EXPONENTIAL = 4;
  public static final Tag[] TAGS_CONVERSION = {
    new Tag(CONVERSION_UNIT, "distance = 1-similarity"),
    new Tag(CONVERSION_LAPLACIAN, "distance=1/(1+similarity)"),
    new Tag(CONVERSION_EXPONENTIAL, "distance=exp(-similarity)")
      };
  /** The method of converting, by default laplacian */
  protected int m_conversionType = CONVERSION_EXPONENTIAL;

  
  /** Construct a vector space from a given set of examples
   * @param strings a list of strings from which the inverted index is
   * to be constructed
   */
  public VectorSpaceMetric() {
    m_stringRefHash = new HashMap();
    m_tokenHash = new HashMap();
    m_stringRefs = new ArrayList();
  }
  
  /** Given a list of strings, build the vector space
   */
  public void buildMetric(List strings) throws Exception {
    m_stringRefHash = new HashMap();
    m_tokenHash = new HashMap();
    // Loop, processing each of the examples
    Iterator stringIterator = strings.iterator();
    while (stringIterator.hasNext()) {
      String string = (String)stringIterator.next();
      // Create a document vector for this document
      HashMapVector vector = m_tokenizer.tokenize(string);
      vector.initLength();
      indexString(string, vector);
    }
    // Now that all strings have been processed, we can calculate the IDF weights for
    // all tokens and the resulting lengths of all weighted document vectors.
    computeIDFandStringLengths();
    System.out.println("Indexed " +  m_stringRefs.size() + " documents with " + size() + " unique terms.");
  }

  /** Index a given string using its corresponding vector */
  protected void indexString(String string, HashMapVector vector) {
    // Create a new reference
    StringReference strRef = new StringReference(string, vector);
    m_stringRefs.add(strRef);
    
    m_stringRefHash.put(string, strRef);
    // Iterate through each of the tokens in the document
    Iterator mapEntries = vector.iterator();
    while (mapEntries.hasNext()) {
      Map.Entry entry = (Map.Entry)mapEntries.next();
      // An entry in the HashMap maps a token to a Weight
      String token = (String)entry.getKey();
      // The count for the token is in the value of the Weight
      int count = (int)((Weight)entry.getValue()).getValue();
      // Add an occurence of this token to the inverted index pointing to this document
      indexToken(token, count, strRef);
    }
  }

  /** Add a token occurrence to the index.
   * @param token The token to index.
   * @param count The number of times it occurs in the document.
   * @param strRef A reference to the String it occurs in.
   */
  protected void indexToken(String token, int count, StringReference strRef) {
    // Find this token in the index
    TokenInfo tokenInfo = (TokenInfo)m_tokenHash.get(token);
    if (tokenInfo == null) {
      // If this is a new token, create info for it to put in the hashtable
      tokenInfo = new TokenInfo();
      m_tokenHash.put(token, tokenInfo);
    }
    // Add a new occurrence for this token to its info
    tokenInfo.occList.add(new TokenOccurrence(strRef, count));
  }

  /** Compute the IDF factor for every token in the index and the length
   * of the string vector for every string referenced in the index. */
  protected void computeIDFandStringLengths() {
    // Let N be the total number of documents indexed
    double N = m_stringRefs.size();
    // Iterate through each of the tokens in the index 
    Iterator mapEntries = m_tokenHash.entrySet().iterator();
    while (mapEntries.hasNext()) {
      // Get the token and the tokenInfo for each entry in the HashMap
      Map.Entry entry = (Map.Entry)mapEntries.next();
      String token = (String)entry.getKey();
      TokenInfo tokenInfo = (TokenInfo)entry.getValue();

      // Get the total number of strings in which this token occurs
      double numStringRefs = tokenInfo.occList.size(); 

      // Calculate the IDF factor for this token
      double idf = Math.log(N/numStringRefs);
      if (idf == 0.0) 
	// If IDF is 0, then just remove this inconsequential token from the index
	mapEntries.remove();
      else {
	tokenInfo.idf = idf;
	// In order to compute document vector lengths,  sum the
	// square of the weights (IDF * occurrence count) across
	// every token occurrence for each document.
	for(int i = 0; i < tokenInfo.occList.size(); i++) {
	  TokenOccurrence occ = (TokenOccurrence)tokenInfo.occList.get(i);
	  if (m_useIDF) { 
	    occ.m_stringRef.m_length = occ.m_stringRef.m_length + Math.pow(idf*occ.m_count, 2);
	  } else {
	    occ.m_stringRef.m_length = occ.m_stringRef.m_length + occ.m_count * occ.m_count;
	  }
	}
      }
    }
    // At this point, every document length should be the sum of the squares of
    // its token weights.  In order to calculate final lengths, just need to
    // set the length of every document reference to the square-root of this sum.
    for(int i = 0; i < m_stringRefs.size(); i++) {
      StringReference stringRef = (StringReference)m_stringRefs.get(i);
      stringRef.m_length = Math.sqrt(stringRef.m_length);
    }
  }
  

  /** Compute similarity between two strings
   * @param s1 first string
   * @param s2 second string
   * @returns similarity between two strings
   */
  public double similarity(String s1, String s2) {
    StringReference stringRef1 = (StringReference) m_stringRefHash.get(s1);
    StringReference stringRef2 = (StringReference) m_stringRefHash.get(s2);
    double length1 = stringRef1.m_length;
    double length2 = stringRef1.m_length;
    HashMapVector v1 = stringRef1.m_vector;
    HashMapVector v2 = stringRef2.m_vector;
    double similarity = 0;

    if (length1 == 0 || length2 == 0) {
      return 0;
    }

    Iterator mapEntries = v1.iterator();
    while (mapEntries.hasNext()) {
      // Get the token and the count for each token in the query
      Map.Entry entry = (Map.Entry)mapEntries.next();
      String token = (String)entry.getKey();
      if (v2.hashMap.containsKey(token)) {
	double count1 = ((Weight)entry.getValue()).getValue();
	double count2 = ((Weight)v2.hashMap.get(token)).getValue();
	TokenInfo tokenInfo = (TokenInfo) m_tokenHash.get(token);

	// add this component unless it was killed (with idf=0)
	if (tokenInfo != null) {
	  double increment = count1 * count2;
	  if (m_useIDF) {
	    increment *= tokenInfo.idf * tokenInfo.idf;
	  }
	  similarity += increment;
	}
      }
    }
    similarity /= length1 * length2;
    return similarity;
  }


  /** The computation of a metric can be either based on distance, or on similarity
   * @returns false because dot product fundamentally computes similarity
   */
  public boolean isDistanceBased() {
    return false;
  }

  
  /** Set the tokenizer to use
   * @param tokenizer the tokenizer that is used
   */
  public void setTokenizer(Tokenizer tokenizer) {
    m_tokenizer = tokenizer;
  }

  /** Get the tokenizer to use
   * @return the tokenizer that is used
   */
  public Tokenizer getTokenizer() {
    return m_tokenizer;
  }

  /** Turn IDF weighting on/off
   * @param useIDF if true, all token weights will be weighted by IDF
   */
  public void setUseIDF(boolean useIDF) {
    m_useIDF = useIDF;
  } 

  /** check whether IDF weighting is on/off
   * @return if true, all token weights are weighted by IDF
   */
  public boolean getUseIDF() {
    return m_useIDF;
  } 


  /** Return the number of tokens indexed.
   * @return the number of tokens indexed*/
  public int size() {
    return m_tokenHash.size();
  }

  /**
   * Returns distance between two strings using the current conversion
   * type (CONVERSION_LAPLACIAN, CONVERSION_EXPONENTIAL, CONVERSION_UNIT, ...)
   * @param string1 First string.
   * @param string2 Second string.
   * @exception Exception if distance could not be estimated.
   */
  public double distance (String string1, String string2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + similarity(string1, string2));
    case CONVERSION_UNIT:
      return 2 * (1 - similarity(string1, string2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-similarity(string1, string2));
    default:
      throw new Exception ("Unknown similarity to distance conversion method");
    }
  }

    /**
   * Set the type of similarity to distance conversion. Values other
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
   * return the type of similarity to distance conversion
   * @return one of CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL
   */
  public SelectedTag getConversionType() {
    return new SelectedTag(m_conversionType, TAGS_CONVERSION);
  }

  /** Create a copy of this metric
   * @return another VectorSpaceMetric with the same exact parameters as this  metric
   */
  public Object clone() {
    VectorSpaceMetric metric = new VectorSpaceMetric();
    metric.setConversionType(new SelectedTag(m_conversionType, TAGS_CONVERSION));
    metric.setTokenizer(m_tokenizer);
    metric.setUseIDF(m_useIDF);
    return metric;
  }

    /**
   * Gets the current settings of NGramTokenizer.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [20];
    int current = 0;

    if (m_conversionType == CONVERSION_EXPONENTIAL) {
      options[current++] = "-E";
    } else if (m_conversionType == CONVERSION_UNIT) {
      options[current++] = "-U";
    }

    if (m_useIDF) {
      options[current++] = "-I";
    }

    options[current++] = "-T";
    options[current++] = Utils.removeSubstring(m_tokenizer.getClass().getName(), "weka.deduping.metrics.");
    if (m_tokenizer instanceof OptionHandler) {
	String[] tokenizerOptions = ((OptionHandler)m_tokenizer).getOptions();
	for (int i = 0; i < tokenizerOptions.length; i++) {
	  options[current++] = tokenizerOptions[i];
	}
      }
    
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -S use stemming
   * -R remove stopwords
   * -N gram size
   */
  public void setOptions(String[] options) throws Exception {
    // TODO
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
