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
 *    KernelVSMetric.java
 *    Copyright (C) 2001 Mikhail Bilenko, Raymond J. Mooney
 *
 */


package weka.deduping.metrics;

import java.util.*;
import java.text.SimpleDateFormat;
import java.io.*;

import weka.core.*;
import weka.deduping.*;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.sparse.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;

/**
 * This class defines a basic string kernel based on vector space
 * Some code borrowed from ir.vsr package by Raymond J. Mooney
 *
 * @author Mikhail Bilenko
 */


public class KernelVSMetric extends StringMetric implements DataDependentStringMetric, LearnableStringMetric,
							    OptionHandler, Serializable {

  /** Strings are mapped to StringReferences in this hash */
  protected HashMap m_stringRefHash = null;

  /** A HashMap where tokens are indexed. Each indexed token maps
   * to a TokenInfo. */
  protected HashMap m_tokenHash = null;

  /** A HashMap where each token is mapped to the corresponding Attribute */
  protected HashMap m_tokenAttrMap = null;

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

  /** The classifier */
  protected DistributionClassifier m_classifier = new SVMlight();

  /** Individual components of the two vectors can be added to the vector-space
   * representation */
  protected boolean m_useIndividualWeights = false;

  /** A special example can be created that contains *all* features so that rare tokens
   * are never ignored (assuming the example will be used as a support vector */
  protected boolean m_useAllFeaturesExample = false; 

  /** has the classifier been trained? */
  protected boolean m_trained = false;

  /** The dataset for the vector space attributes */
  protected Instances m_instances = null;

  
  /** Construct a vector space from a given set of examples
   * @param strings a list of strings from which the inverted index is
   * to be constructed
   */
  public KernelVSMetric() {
    m_stringRefHash = new HashMap();
    m_tokenHash = new HashMap();
    m_stringRefs = new ArrayList();
  }
  
  /** Given a list of strings, build the vector space
   */
  public void buildMetric(List strings) throws Exception {
    m_stringRefHash = new HashMap();
    m_tokenHash = new HashMap();
    m_stringRefs = new ArrayList();
    m_trained = false;
    
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
    initKernel();
    System.out.println("Indexed " +  m_stringRefs.size() + " strings with " + size() + " unique terms.");
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
	// If IDF is 0, then just remove this "omnipresent" token from the index
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

  /** Provided that all features are known, initialize the feature space for the kernel
   */
  protected void initKernel() {
    m_tokenAttrMap = new HashMap();
    
    // create the features
    FastVector attrVector = new FastVector(m_tokenHash.size());
    Iterator iterator = m_tokenHash.keySet().iterator();
    while (iterator.hasNext()) {
      String token = (String) iterator.next();
      Attribute attr = new Attribute(token);
      attrVector.addElement(attr);
      m_tokenAttrMap.put(token, attr);
    }

    // If we are interested in a "concatenated" representation, add the extra features
    if (m_useIndividualWeights) {
      Iterator iterator1 = m_tokenHash.keySet().iterator();
      while (iterator1.hasNext()) {
	String token = (String) iterator1.next();
	Attribute attr_s1 = new Attribute("s1_" + token);
	Attribute attr_s2 = new Attribute("s2_" + token);
	attrVector.addElement(attr_s1);
	attrVector.addElement(attr_s2);
	m_tokenAttrMap.put("s1_" + token, attr_s1);
	m_tokenAttrMap.put("s2_" + token, attr_s2);
      } 
    } 

    // create the class attribute
    FastVector classValues = new FastVector();
    classValues.addElement("pos");
    classValues.addElement("neg");
    Attribute classAttr = new Attribute("__class__", classValues);
    attrVector.addElement(classAttr);

    // create the dataset for the vector space
    m_instances = new Instances("diffInstances", attrVector, 3000);
    m_instances.setClass(classAttr);
  } 

  
  /** Train the metric given a set of aligned strings
   * @param pairList the training data as a list of StringPair's
   * @returns distance between two strings
   */
  public void trainMetric(ArrayList pairList) throws Exception {
    m_instances.delete();

    // some training pairs will be deemed unworthy
    int numDiscardedPositives = 0;
    int numDiscardedNegatives = 0;
    
    // populate the training instances
    HashSet seenInstances = new HashSet();
    for (int i = 0; i < pairList.size(); i++) {
      StringPair pair = (StringPair) pairList.get(i);
      SparseInstance pairInstance = createPairInstance(pair.str1, pair.str2);
      double[] values = pairInstance.toDoubleArray();
      if (seenInstances.contains(values)) {
	System.out.println("Seen instance vector, skipping: " + pairInstance + "   <= " + pair.str1 + "\t" + pair.str2);
      } else { 
	// this pair vector has not been seen before
	boolean good = true;

	// set the dataset and the class value
	pairInstance.setDataset(m_instances);
	if (pair.positive) {
	  pairInstance.setClassValue(0);
	  if (pairInstance.numValues() < 1) {
	    System.out.println("Too few values, skipping: " + pairInstance + "   <= " + pair.str1 + "\t" + pair.str2);
	    good = false;
	    numDiscardedPositives++;
	  }
	} else {
	  // negative example
	  pairInstance.setClassValue(1);
	}
	
	if (good) {
	  m_instances.add(pairInstance);
	}
      }
    }
    System.out.println("Discarded " + numDiscardedPositives + " positives; " + 
		       "went from " + pairList.size() + " down to " + m_instances.numInstances() + " training instances");

    // Add an artificial example containing all features to prevent rare features from being excluded
    if (m_useAllFeaturesExample) { 
      Instance allFeaturesInstance = new Instance(m_instances.numAttributes()); 
      allFeaturesInstance.setDataset(m_instances);
      allFeaturesInstance.setClassValue(0);
      Iterator mapEntries = m_tokenHash.entrySet().iterator();
      while (mapEntries.hasNext()) {
	Map.Entry entry = (Map.Entry)mapEntries.next();
	String token = (String)entry.getKey();
	TokenInfo tokenInfo = (TokenInfo)entry.getValue();
	Attribute attr = (Attribute) m_tokenAttrMap.get(token);
	allFeaturesInstance.setValue(attr, tokenInfo.idf);

	// if we are using concatenated representation, add those features as well
	if (m_useIndividualWeights) {
	  Attribute attr1 = (Attribute) m_tokenAttrMap.get("s1_" + token);
	  allFeaturesInstance.setValue(attr1, tokenInfo.idf);
	  Attribute attr2 = (Attribute) m_tokenAttrMap.get("s2_" + token);
	  allFeaturesInstance.setValue(attr2, tokenInfo.idf);
	} 
      }
      normalizeInstance(allFeaturesInstance);
      m_instances.add(allFeaturesInstance);

      if (m_classifier instanceof SVMcplex) {
	((SVMcplex)m_classifier).setUseAllFeaturesExample(true); 
      } 
    }
    
    

    // BEGIN SANITY CHECK
    // dump diff-instances into a temporary file
    if (false) { 
      try {
	Instances instances = new Instances(m_instances);

	// dump instances
	File diffDir = new File("/tmp/KVS");
	diffDir.mkdir();
	String diffName = Utils.removeSubstring(m_classifier.getClass().getName(), "weka.classifiers.");
	PrintWriter writer = new PrintWriter(new BufferedOutputStream (new FileOutputStream(diffDir.getPath() + "/" +
											    diffName + ".arff")));
	writer.println(instances.toString());
	writer.close();

	// Do a sanity check - dump out the diffInstances, and
	// evaluation classification with an SVM. 
	long trainTimeStart = System.currentTimeMillis();
	//	SVMlight classifier = new SVMlight();

	Classifier classifier = (Classifier) Class.forName(m_classifier.getClass().getName()).newInstance();
	if (m_classifier instanceof OptionHandler) { 
	  ((OptionHandler)classifier).setOptions(((OptionHandler)m_classifier).getOptions());
	}
	
	Evaluation eval = new Evaluation(instances);
	eval.crossValidateModel(classifier, instances, 2);
	writer = new PrintWriter(new BufferedOutputStream (new FileOutputStream(diffDir.getPath() + "/" +
										diffName + ".dat", true)));
	writer.println(eval.pctCorrect());
	writer.close();
	System.out.println("** String sanity: " + (System.currentTimeMillis() - trainTimeStart) + " ms; " +
			   eval.pctCorrect() + "% correct\t" +
			   eval.numFalseNegatives(0) + "(" + eval.falseNegativeRate(0) + "%) false negatives\t" +
			   eval.numFalsePositives(0) + "(" + eval.falsePositiveRate(0) + "%) false positives\t");
      } catch (Exception e) {
	e.printStackTrace();
	System.out.println(e.toString()); 
      }
    }
    // END SANITY CHECK
    System.out.println((new SimpleDateFormat("HH:mm:ss:")).format(new Date()) +
		       weka.classifiers.sparse.IBkMetric.concatStringArray(((OptionHandler)m_classifier).getOptions()));

    System.out.println("Now got " + m_instances.numInstances());
    m_classifier.buildClassifier(m_instances);
    m_trained = true;
  }

  /** Given a pair of strings and a label (same-class/different-class),
   * create a diff-instance
   */
  protected SparseInstance createPairInstance(String s1, String s2) {
    StringReference stringRef1 = (StringReference) m_stringRefHash.get(s1);
    StringReference stringRef2 = (StringReference) m_stringRefHash.get(s2);
    double invLength = 1/(stringRef1.m_length * stringRef2.m_length);
    HashMapVector v1 = stringRef1.m_vector;
    HashMapVector v2 = stringRef2.m_vector;
    SparseInstance pairInstance = new SparseInstance(1, new double[0], new int[0], m_tokenHash.size()+1);

    // calculate all the components of the kernel
    Iterator mapEntries = v1.iterator();
    while (mapEntries.hasNext()) {
      Map.Entry entry = (Map.Entry)mapEntries.next();
      String token = (String)entry.getKey();
      if (v2.hashMap.containsKey(token)) {
	Attribute attr = (Attribute) m_tokenAttrMap.get(token);
	double tf1 = ((Weight)entry.getValue()).getValue();
	double tf2 = ((Weight)v2.hashMap.get(token)).getValue();
	TokenInfo tokenInfo = (TokenInfo) m_tokenHash.get(token);
	// add this component unless it was killed (with idf=0)
	if (tokenInfo != null) {
	  if (m_useIDF) { 
	    pairInstance.setValue(attr, tf1 * tf2 * tokenInfo.idf * tokenInfo.idf * invLength );
	  } else {
	    pairInstance.setValue(attr, tf1 * tf2 * invLength );
	  }

	  if (m_useIndividualWeights) {
	    Attribute attr_s1 = (Attribute) m_tokenAttrMap.get("s1_" + token);
	    Attribute attr_s2 = (Attribute) m_tokenAttrMap.get("s2_" + token);

	    if (m_useIDF) {  // TODO:  this is not right; invLength should be different!
	      pairInstance.setValue(attr_s1, tf1 * tokenInfo.idf * invLength);
	      pairInstance.setValue(attr_s2, tf2 * tokenInfo.idf * invLength);
	    } else {
	      pairInstance.setValue(attr_s1, tf1 * invLength);
	      pairInstance.setValue(attr_s2, tf2 * invLength);
	    } 
	  } 
	}
      }
    }
    
    return pairInstance;
  }
  

  /** Compute similarity between two strings
   * @param s1 first string
   * @param s2 second string
   * @returns similarity between two strings
   */
  public double similarity(String s1, String s2) throws Exception {
    SparseInstance pairInstance = createPairInstance(s1, s2);
    pairInstance.setDataset(m_instances);
    double sim = 0;

    // if the classifier has been trained, use it.
    if (m_trained) {
      double[] res = m_classifier.distributionForInstance(pairInstance);
      sim = res[0];
    } else {
      // otherwise, return the old-fashioned dot product
      for (int j = 0; j < pairInstance.numValues(); j++) {
	Attribute attribute = pairInstance.attributeSparse(j);
	int attrIdx = attribute.index();
	sim += pairInstance.value(attrIdx);
      }
    } 
    return sim;
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

  /**
   * Set the classifier
   *
   * @param classifier the classifier
   */
  public void setClassifier (DistributionClassifier classifier) {
    m_classifier = classifier;
  }

  /**
   * Get the classifier
   *
   * @returns the classifierthat this metric employs
   */
  public DistributionClassifier getClassifier () {
    return m_classifier;
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

  /** Turn using individual components on/off
   * @param useIndividualStrings if true, individual token weghts are included in the pairwise representation  */
  public void setUseIndividualStrings(boolean useIndividualStrings) {
    m_useIndividualWeights = useIndividualStrings;
  } 

  /** Turn using individual components on/off
   * @return true if individual token weights are included in the pairwise representation */
  public boolean getUseIndividualStrings() {
    return m_useIndividualWeights;
  } 


  /** Turn adding a special all-features example on/off
   * @param useAllFeaturesExample if true, a special training example will be constructed that incorporates all features */
  public void setUseAllFeaturesExample(boolean useAllFeaturesExample) {
    m_useAllFeaturesExample = useAllFeaturesExample;
  } 

  /** Check whether a special all-features example is being added
   * @return  true if a special training example will be constructed that incorporates all features */
  public boolean getUseAllFeaturesExample() {
    return m_useAllFeaturesExample;
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
   * @return another KernelVSMetric with the same exact parameters as this  metric
   */
  public Object clone() {
    KernelVSMetric metric = new KernelVSMetric();
    metric.setConversionType(new SelectedTag(m_conversionType, TAGS_CONVERSION));
    metric.setTokenizer(m_tokenizer);
    metric.setUseIDF(m_useIDF);
    metric.setUseIndividualStrings(m_useIndividualWeights);
    metric.setUseAllFeaturesExample(m_useAllFeaturesExample);
    try {
      DistributionClassifier classifier = (DistributionClassifier) Class.forName(m_classifier.getClass().getName()).newInstance();
      if (m_classifier instanceof OptionHandler) { 
	((OptionHandler)classifier).setOptions(((OptionHandler)m_classifier).getOptions());
      }
      metric.setClassifier(classifier);
    } catch (Exception e) {
      System.err.println("Problems cloning metric " + this.getClass().getName() + ": " + e.toString());
      e.printStackTrace();
      System.exit(1);
    }
    return metric;
  }

  /**
   * Gets the current settings of NGramTokenizer.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [40];
    int current = 0;

    if (m_conversionType == CONVERSION_EXPONENTIAL) {
      options[current++] = "-E";
    } else if (m_conversionType == CONVERSION_UNIT) {
      options[current++] = "-U";
    }

    if (m_useAllFeaturesExample) {
      options[current++] = "-AF";
    } 

    if (m_useIDF) {
      options[current++] = "-I";
    }

    if (m_useIndividualWeights) {
      options[current++] = "-V";
    }

    options[current++] = "-T";
    options[current++] = Utils.removeSubstring(m_tokenizer.getClass().getName(), "weka.deduping.metrics.");
    if (m_tokenizer instanceof OptionHandler) {
      String[] tokenizerOptions = ((OptionHandler)m_tokenizer).getOptions();
      for (int i = 0; i < tokenizerOptions.length; i++) {
	options[current++] = tokenizerOptions[i];
      }
    }
    
    options[current++] = "-C";
    options[current++] = Utils.removeSubstring(m_classifier.getClass().getName(), "weka.classifiers.");
    if (m_classifier instanceof OptionHandler) {
      String[] classifierOptions = ((OptionHandler)m_classifier).getOptions();
      for (int i = 0; i < classifierOptions.length; i++) {
	options[current++] = classifierOptions[i];
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

  /** Given an instance, normalize it to be a unit vector. Destructive!
   * @param instance instance to be normalized
   */
  protected void normalizeInstance(Instance instance) { 
    double norm = 0;
    double values [] = instance.toDoubleArray();
    for (int i=0; i < values.length; i++) {
      if (i != instance.classIndex()) { // don't normalize the class index 
	norm += values[i] * values[i];
      }
    }
    norm = Math.sqrt(norm);
    if (norm != 0) { 
      for (int i=0; i<values.length; i++) {
	if (i != instance.classIndex()) { // don't normalize the class index 
	  values[i] /= norm;
	}
      }
      instance.setValueArray(values);
    }
  }    
}
