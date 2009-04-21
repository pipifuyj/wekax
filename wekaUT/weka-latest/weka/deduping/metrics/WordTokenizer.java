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
 *    WordTokenizer.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */


package weka.deduping.metrics;

import java.util.*;
import java.io.*;

import weka.core.*;

/**
 * This class defines a tokenizer that turns strings into HashMapVectors
 * using the native Java StringTokenizer
 *
 * @author Mikhail Bilenko
 */
public class WordTokenizer extends Tokenizer implements Serializable, OptionHandler, Cloneable {
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

  /** A default set of delimiters */
  protected String m_delimiters = " \t\n\r\f\'\"\\!@#$%^&*()_-+={}<>,.;:|[]{}/*~`";
  
  /** The default minimum length of a token */
  protected int m_minTokenLength = 1;

  /** A default constructor */
  public WordTokenizer() {
    super();
    setStemming(m_stemming);
    setStopwordRemoval(m_stopwordRemoval); 
  } 

  /** Take a string and create a vector of tokens from it
   * @param string a String to tokenize
   * @returns vector with individual tokens
   */
  public HashMapVector tokenize(String string) {
    StringTokenizer tokenizer = new StringTokenizer(string, m_delimiters);
    HashMapVector result = new HashMapVector();

    if (m_stopwordRemoval && m_stopwordSet == null) {
      setStopwordRemoval(true);
    } 
    
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      if (token.length() >= m_minTokenLength) {
	if (m_caseInsensitive) {
	  token = token.toLowerCase();
	}
	if (m_stemming) {
	  token = stem(token);
	}
	if (m_stopwordRemoval) {
	  if (!m_stopwordSet.contains(token)) {
	    result.increment(token);
	  }
	} else {
	  result.increment(token);
	} 
      } 
    }
    return result;
  }

  /** Take a string and create a TokenString from it
   * @param string a String to tokenize
   * @returns a TokenString composed of individual tokens
   */
  public TokenString getTokenString(String string) {
    StringTokenizer tokenizer = new StringTokenizer(string, m_delimiters);
    TokenString ts = new TokenString(string);
    ArrayList tokenList = new ArrayList();

    if (m_stopwordRemoval && m_stopwordSet == null) {
      setStopwordRemoval(true);
    } 

    if (m_caseInsensitive) {
      string = string.toLowerCase();
    }
    
    while (tokenizer.hasMoreTokens()) {
      String token = tokenizer.nextToken();
      if (token.length() >= m_minTokenLength) {
	if (m_stemming) {
	  System.out.print(token + "->");
	  token = stem(token);
	  System.out.println(token);
	}

	if (!m_stopwordRemoval || !m_stopwordSet.contains(token)) {
	  Object o = m_stringIDmap.get(token);
	  if (o == null) {
	    m_stringIDmap.put(token, new Integer(m_currIDidx++));
	  }
	  tokenList.add(token);
	}
      } 
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


  /** Specify which delimiters to use
   * @param delim a string containing delmiters to use
   */
  public void setDelimiters(String delimiters) {
    m_delimiters = new String(delimiters);
  } 

  /** Get the delimiters 
   * @return a string containing delmiters that are used
   */
  public String getDelimiters() {
    return m_delimiters;
  }

  /** Set the minimum token length
   * @param minTokenLength the minimum length of a token
   */
  public void setMinTokenLength(int minTokenLength) {
    m_minTokenLength = minTokenLength;
  }

  /** Get the minimum token length
   * @return the minimum length of a token
   */
  public int getMinTokenLength() {
    return m_minTokenLength;
  }

  /**
   * Gets the current settings of WordTokenizer.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [10];
    int current = 0;

    if (m_caseInsensitive) {
      options[current++] = "-I";
    } 
    
    if (m_stemming) {
      options[current++] = "-S";
    }

    if (m_stopwordRemoval) {
      options[current++] = "-R";
    }
    
    options[current++] = "-m";
    options[current++] = "" + m_minTokenLength;
    
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
   * -m minimum length of a token for it to be included
   */
  public void setOptions(String[] options) throws Exception {
    System.out.println("Inside setOPtions + " + options.length); 
    setStemming(Utils.getFlag('S', options));

    setStopwordRemoval(Utils.getFlag('R', options));
    
    String minTokenLengthString = Utils.getOption('m', options);
    if (minTokenLengthString.length() != 0) {
      setMinTokenLength(Integer.parseInt(minTokenLengthString));
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
    
    newVector.addElement(new Option("\tMinimum length of token for it to be included\n",
				    "m", 1, "-m"));
    
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
	m_stopwordSet = new HashSet();
	BufferedReader in = new BufferedReader(new FileReader(m_stopwordFilename));
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
    if (m_stopwordSet != null)   System.out.println("Size of the hash: " + m_stopwordSet.size());
    return m_stopwordRemoval;
  }
}
