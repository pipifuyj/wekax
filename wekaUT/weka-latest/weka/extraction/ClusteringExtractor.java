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
 *    ClusteringExtractor.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.extraction;

import weka.core.*;
import weka.clusterers.*;
import java.util.*;

/** An abstract extractor class. Takes a set of objects and trains on it;
 * then can be used for extraction on a testing set.
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */
public class ClusteringExtractor extends Extractor implements  OptionHandler {

  /** The baseline extractor that is used */
  protected Extractor m_extractor = null;   // TODO:  fill in some basic extractor

  /** The clusterer */
  protected Clusterer m_clusterer = new SeededKMeans();

  /** Two fundamental modes.
   * We can either cluster documents, and train separate extractors
   * depending what the document is like
   * Or, we can cluster text segments and train separate extractors
   * for different segments
   * Or, we could mix, but we're not touching this for now...
   */
  public static final int MODE_DOCUMENT_CLUSTERS = 1;
  public static final int MODE_SEGMENT_CLUSTERS = 2;
  public static final int MODE_MIXED = 4;
  public static final Tag[] TAGS_CLUSTERING_MODE = {
    new Tag(MODE_DOCUMENT_CLUSTERS, "Document clusters"),
    new Tag(MODE_SEGMENT_CLUSTERS, "Text segment clusters"),
    new Tag(MODE_MIXED, "Both document and segment clusters")
  };
  protected int m_mode = MODE_DOCUMENT_CLUSTERS;

  /** Verbose? */
  protected boolean m_verbose = false;


  
  
  /** A default constructor */
  public ClusteringExtractor() {
  } 
  
  /** Given training data, train the extractor
   * @param labeledData a set of training data
   * @param unlabeledData we don't plan to use transduction here for now
   */
  public void trainExtractor(Instances labeledData, Instances unlabeledData) throws Exception{
    switch(m_mode) {

    case MODE_DOCUMENT_CLUSTERS:
      //  1. cluster labeledData
      //  2. train an extractor for each cluster
      break;

    case MODE_SEGMENT_CLUSTERS:
      //  1. segment each document and populate an Instances object with segments
      //  2. train an extractor for each cluster
      break;

    case MODE_MIXED:
      System.err.println("Mixed mode not implemented for now");
    } 
  }

  /** Perform extraction on a set of data. 
   * @param testData a set of instances on which to perform extraction
   * @param docFillerMap a map where the uniqueID of an instance (document) is mapped to a
   * HashMap, which maps fillers to a list of Integer positions
   */
  public void testExtractor(Instances testData, HashMap docFillerMap) throws Exception {
    switch(m_mode) {

    case MODE_DOCUMENT_CLUSTERS:
      for (int i = 0; i < testData.numInstances(); i++) {
	Instance instance = testData.instance(i);
	// 1. assign instance to a cluster
	// 2. apply that cluster's  extractor to get the result
      }
      break;
      
    case MODE_SEGMENT_CLUSTERS:
      for (int i = 0; i < testData.numInstances(); i++) {
	Instance instance = testData.instance(i);
	// 1. segment instance
	// 2. assign each segment to a cluster
	// 3. apply that cluster's extractor to get the result
      }
      break;
      
    case MODE_MIXED:
      System.err.println("Mixed mode not implemented for now");
    } 
  }


  /** Set the clustering mode 
   * @param mode one of MODE_DOCUMENT_CLUSTERS or MODE_SEGMENT_CLUSTERS
   */
  public void setMode(SelectedTag mode) {
    if (mode.getTags() == TAGS_CLUSTERING_MODE) {
      m_mode = mode.getSelectedTag().getID();
    }
  }

  /**
   * return the clustering mode
   * @return one of MODE_DOCUMENT_CLUSTERS or MODE_SEGMENT_CLUSTERS
   */
  public SelectedTag getMode() {
    return new SelectedTag(m_mode, TAGS_CLUSTERING_MODE);
  }


  /** Set the clusterer
   * @param clusterer the clusterer to be used
   */
  public void setClusterer(Clusterer clusterer) {
    m_clusterer = clusterer;
  } 


  /** Get the clusterer
   * @return the clusterer that is used
   */
  public Clusterer getClusterer() {
    return m_clusterer;
  }

  /** Set the extractor
   * @param extractor the extractor to be used
   */
  public void setExtractor(Extractor extractor) {
    m_extractor = extractor;
  } 
  
  /** Get the extractor
   * @return the extractor that is used
   */
  public Extractor getExtractor() {
    return m_extractor;
  }

  /**
   * set the verbosity level of the clusterer
   * @param verbose messages on(true) or off (false)
   */
  public void setVerbose (boolean verbose) {
    m_verbose = verbose;
  }

  /**
   * get the verbosity level of the clusterer
   * @return messages on(true) or off (false)
   */
  public boolean getVerbose () {
    return m_verbose;
  }

  
  /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   **/
  public Enumeration listOptions() {
    
    Vector newVector = new Vector(0);
    
    // TODO:  list options... last thing we care about for now
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   *
   * Valid options are:<p>
   *
   * -D document-clustering mode
   * or
   * -S segment-clustering mode
   *
   * -E extractor-name extractor-options <br>
   * extractor and its options
   *
   * -C clusterer-name clusterer-options <br>
   * clusterer and its options <p>
   *
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   *
   **/
  public void setOptions(String[] options) throws Exception {
    String optionString;

    // get the mode
    if (Utils.getFlag('D', options)) {
      setMode(new SelectedTag(MODE_DOCUMENT_CLUSTERS, TAGS_CLUSTERING_MODE));
    } else if (Utils.getFlag('S', options)) {
      setMode(new SelectedTag(MODE_SEGMENT_CLUSTERS, TAGS_CLUSTERING_MODE));
    } else {
      throw new Exception("Must specify -D or -S for clustering mode");
    }

    // get the extractor specification
    optionString = Utils.getOption('E', options);
    if (optionString.length() != 0) {
      String[] extractorSpec = Utils.splitOptions(optionString);
      String extractorName = extractorSpec[0]; 
      extractorSpec[0] = "";
      if (m_verbose) {
	System.out.println("Extractor name: " + extractorName + "\nExtractor parameters: " + concatStringArray(extractorSpec));
      }
      setExtractor(Extractor.forName(extractorName, extractorSpec));
    }

    // get the clusterer specification
     optionString = Utils.getOption('E', options);
    if (optionString.length() != 0) {
      String[] clustererSpec = Utils.splitOptions(optionString);
      String clustererName = clustererSpec[0]; 
      clustererSpec[0] = "";
      if (m_verbose) {
	System.out.println("Clusterer name: " + clustererName + "\nClusterer parameters: " + concatStringArray(clustererSpec));
      }
      setClusterer(Clusterer.forName(clustererName, clustererSpec));
    }
  }

    /** A little helper to create a single String from an array of Strings
   * @param strings an array of strings
   * @returns a single concatenated string, separated by commas
   */
  public static String concatStringArray(String[] strings) {
    String result = new String();
    for (int i = 0; i < strings.length; i++) {
      result = result + "\"" + strings[i] + "\" ";
    }
    return result;
  } 



  /**
   * Gets the current settings of Greedy Agglomerative Clustering
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    
    String [] options = new String [70];
    int current = 0;

    if (m_mode == MODE_DOCUMENT_CLUSTERS) {
      options[current++] = "-D";
    } else if (m_mode == MODE_SEGMENT_CLUSTERS) {
      options[current++] = "-S";
    } 

    // the extractor name and options
    options[current++] = "-E";
    options[current++] = Utils.removeSubstring(m_extractor.getClass().getName(), "weka.extraction.");;
    if (m_extractor instanceof OptionHandler) {
      String[] extractorOptions = ((OptionHandler)m_extractor).getOptions();
      for (int i = 0; i < extractorOptions.length; i++) {
	options[current++] = extractorOptions[i];
      }
    }

    // the clusterer name and options
    options[current++] = "-C";
    options[current++] = Utils.removeSubstring(m_clusterer.getClass().getName(), "weka.clusterers.");
    if (m_clusterer instanceof OptionHandler) {
      String[] clustererOptions = ((OptionHandler)m_clusterer).getOptions();
      for (int i = 0; i < clustererOptions.length; i++) {
	options[current++] = clustererOptions[i];
      }
    } 

    // fill the rest with blanks
    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }
  

}

