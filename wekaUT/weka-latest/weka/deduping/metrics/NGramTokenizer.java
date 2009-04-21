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
 *    NGramTokenizer.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */


package weka.deduping.metrics;

import java.util.*;
import java.io.*;
import weka.core.*;

/**
 * This class defines a tokenizer that turns strings into HashMapVectors 
 * of n-grams
 *
 * @author Mikhail Bilenko
 */
public class NGramTokenizer extends Tokenizer implements Serializable, OptionHandler {
  /** Converting all tokens to lowercase */
  protected boolean m_caseInsensitive = true;

  /** Stemming */
  protected boolean m_stemming = false;
  protected Porter m_stemmer = new Porter();

  /** Stopword removal */
  protected boolean m_stopwordRemoval = false;
  /** The with the stopword list */
  protected static String m_stopwordFilename = "/u/mbilenko/weka/weka/deduping/metrics/stopwords.txt";
  /** Stopword hash */
  protected static HashSet m_stopwordSet = null;

  /** Length of an n-gram */
  protected int m_n = 3;

  /** A default set of space-equivalent characters */
  protected String m_spaceEquivalents = "\t\n\r\f\'\"\\!@#$%^&*()_-+={}<>,.;:|[]{}/*~`";
  protected char[] m_spaceChars = null;

  /** if true, all space equivalents will be replaced with a single space */ 
  protected boolean m_replaceSpaces = false; 


  /** A default constructor */
  public NGramTokenizer() {
    super();
    m_spaceChars = m_spaceEquivalents.toCharArray();
    setStemming(false);
    setStopwordRemoval(false);
  } 

  /** Take a string and create a vector of n-gram tokens from it
   * @param string a String to tokenize
   * @returns vector with individual tokens
   */
  public HashMapVector tokenize(String string) {
    if (m_caseInsensitive) {
      string = string.toLowerCase();
    }
    
    StringBuffer filteredString = new StringBuffer();

    // only need to tokenize if stemming, or removing stopwords, or replacing space equivalents
    if (m_stemming || m_stopwordRemoval || m_replaceSpaces) {
      StringTokenizer tokenizer = new StringTokenizer(string, m_spaceEquivalents, true);
      while (tokenizer.hasMoreTokens()) {
	String token = tokenizer.nextToken();

	if (m_stemming) {
	  token = stem(token);
	}
	if (m_stopwordRemoval && m_stopwordSet.contains(token)) {
	  continue;
	}

	if (m_replaceSpaces && token.length() == 1) {
	  if (m_spaceEquivalents.indexOf(token) > -1) {
	    filteredString.append(" ");
	  }
	} else {
	  filteredString.append(token);
	}
      }
    } else {
      filteredString = new StringBuffer(string);
    } 
    
    char[] chars = filteredString.toString().toCharArray();
   
    HashMapVector result = new HashMapVector();

    for (int i = 0; i < chars.length - m_n; i++) {
      String token = new String(chars, i, m_n);
      result.increment(token);
    } 
    return result;
  }

  /** Take a string and create a TokenString of overlapping n-gram tokens from it
   * @param string a String to tokenize
   * @returns vector with individual tokens
   */
  public TokenString getTokenString(String string) {
    TokenString ts = new TokenString(string);
    ArrayList tokenList = new ArrayList();
    
    StringBuffer filteredString = new StringBuffer();
    // only need to tokenize if stemming, or removing stopwords, or replacing space equivalents
    if (m_stemming || m_stopwordRemoval || m_replaceSpaces) {
      StringTokenizer tokenizer = new StringTokenizer(string, m_spaceEquivalents, true);
      while (tokenizer.hasMoreTokens()) {
	String token = tokenizer.nextToken();

	if (m_stemming) {
	  token = stem(token);
	}
	if (m_stopwordRemoval && m_stopwordSet.contains(token)) {
	  continue;
	}

	if (m_replaceSpaces && token.length() == 1) {
	  if (m_spaceEquivalents.indexOf(token) > -1) {
	    filteredString.append(" ");
	  }
	} else {
	  filteredString.append(token);
	}
      }
    } else {
      filteredString = new StringBuffer(string);
    } 
    
    char[] chars = filteredString.toString().toCharArray();
   
    HashMapVector result = new HashMapVector();

    for (int i = 0; i < chars.length - m_n; i++) {
      String token = new String(chars, i, m_n);
      Object o = m_stringIDmap.get(token);
      if (o == null) {
	m_stringIDmap.put(token, new Integer(m_currIDidx++));
      }
      tokenList.add(token);
    }

    // convert the tokenList into the two arrays inside TokenString
    ts.tokens = new String[tokenList.size()];
    ts.tokens = (String[]) tokenList.toArray(ts.tokens);
    ts.tokenIDs = new int[ts.tokens.length];
    for (int i = 0; i < ts.tokens.length; i++) {
      ts.tokenIDs[i] = ((Integer)m_stringIDmap.get(ts.tokens[i])).intValue();
    } 
    return ts;
  }


  /** Set the gram length
   * @param n the gram length
   */
  public void setN(int n) {
    m_n = n;
  }

  /** Get the gram length
   * @return the gram length
   */
  public int getN() {
    return m_n;
  }

  
  /** Specify which characters should be treated as spaces
   * @param spaceEquivalents a string containing space equivalents
   */
  public void setSpaceEquivalents(String spaceEquivalents) {
    m_spaceEquivalents = new String(spaceEquivalents);
    m_spaceChars = m_spaceEquivalents.toCharArray();
  } 

  /** Get the haracters that should be treated as spaces 
   * @return a string containing space equivalents
   */
  public String getSpaceEquivalents() {
    return m_spaceEquivalents;
  }


  /** Turn on/off replacing space equivalents with a single space */
  public void setReplaceSpaces(boolean replaceSpaces) {
    m_replaceSpaces = replaceSpaces;
  }
  public boolean getReplaceSpaces() {
    return m_replaceSpaces;
  } 

  
  /**
   * Gets the current settings of NGramTokenizer.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [10];
    int current = 0;

    if (m_stemming) {
      options[current++] = "-S";
    }

    if (m_stopwordRemoval) {
      options[current++] = "-R";
    }

    if (m_replaceSpaces) {
      options[current++] = "-spaces";
    }
    
    options[current++] = "-N";
    options[current++] = "" + m_n;
    
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
    setStemming(Utils.getFlag('S', options));

    setStopwordRemoval(Utils.getFlag('R', options));
    
    
    String nString = Utils.getOption('N', options);
    if (nString.length() != 0) {
      setN(Integer.parseInt(nString));
    }
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector(5);

    newVector.addElement(new Option("\tUse Porter stemmer for stemming\n",
				    "S", 0, "-S"));

    newVector.addElement(new Option("\tRemove stopwords\n",
				    "R", 0, "-R"));
    
    newVector.addElement(new Option("\tGram size\n",
				    "N", 1, "-N"));
    
    return newVector.elements();
  }

    /** Turn case sensitivity on/off
   * @param caseInsensitive if true, the tokenizer is case-insensitive
   */
  public void setCaseInsensitive(boolean caseInsensitive) { 
    m_caseInsensitive = caseInsensitive;
  }

  /** Turn case sensitivity on/off
   * @return if true, the tokenizer is case-insensitive
   */
  public boolean getCaseInsensitive() { 
    return m_caseInsensitive;
  }

   /** Turn stemming on/off
   * @param stemming if true, stemming is used
   */
  public void setStemming(boolean stemming) { 
    m_stemming = stemming;
    if (stemming) {
      m_stemmer = new Porter();
    }
  }

  /** Find out whether stemming is on/off
   * @return if true, stemming is used
   */
  public boolean getStemming() { 
    return m_stemming;
  }

  /** Stem a given token
   * @param token the token to be stemmed
   * @return a new token resulting from applying the stemmer
   */
  public String stem(String token) {
    return m_stemmer.stripAffixes(token);
  }
  
  /** Turn stopword removal on/off and load the stopwords
   * @param stopwordRemoval if true, stopwords from m_stopwordFile will be removed
   */
  public void setStopwordRemoval(boolean stopwordRemoval) { 
    m_stopwordRemoval = stopwordRemoval;
    if (m_stopwordRemoval) {
      try {
	BufferedReader in = new BufferedReader(new FileReader(m_stopwordFilename));
	m_stopwordSet = new HashSet();
	String stopword;
	while ((stopword = in.readLine()) != null) {
	  m_stopwordSet.add(stopword);
	}
      } catch (Exception e) {
	System.out.println("Problems initializing the stopwords from " + m_stopwordFilename);
      }
    }
  }

  /** Get whether stopword removal is on or off
   * @return true if stopword removal is on
   */
  public boolean getStopwordRemoval() {
    return m_stopwordRemoval;
  }
}

