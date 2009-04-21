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
 *    BasicDeduper.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.deduping;

import weka.core.*;
import weka.deduping.metrics.*;
import weka.deduping.blocking.*;

import java.io.Serializable;
import java.util.*;

import weka.clusterers.Cluster;

/** A basic deduper class that takes a set of objects and
 * identifies disjoint subsets of duplicates
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.7 $
 */
public class BasicDeduper extends Deduper implements OptionHandler, Serializable {

  /** A metric measuring similarity between every pair of instances */
  InstanceMetric m_metric = new SumInstanceMetric();

  /** The proportion of the training fold that should be used for training*/
  protected double  m_trainProportion = 1.0;
  
  /** distance matrix containing the distance between each pair */
  protected double[][] m_distanceMatrix = null;
  
  /** instance hash, where each Integer index is hashed to an instance */
  protected HashMap m_instancesHash = null;

  /** reverse instance hash, where each instance is hashed to its Integer index */
  protected HashMap m_reverseInstancesHash = null;

  /** the attribute indeces on which to do deduping */
  protected int[] m_attrIdxs = null;

  /** The total number of true objects */
  protected int m_numObjects = 0;
  
  /** An array containing class values for instances (for faster statistics) */
  protected double[] m_classValues = null;

  /** Number of clusters in the process*/
  protected int m_numCurrentObjects = 0;

  /** holds the clusters */
  protected ArrayList m_clusters = null;

  /** A set of instances to dedupe */
  protected Instances m_testInstances = null;

  /** Use blocking ? */
  protected boolean m_useBlocking = false; 
  
  /**
   * temporary variable holding cluster assignments
   */
  protected int [] m_clusterAssignments;

  /** verbose? */
  protected boolean m_debug = false;

  /** Statistics */
  protected int m_numTotalPairs = 0;
  protected int m_numGoodPairs = 0;
  protected int m_numTruePairs = 0;

  protected int m_numTotalPairsTrain = 0;  // the overall number of pairs in the test split
  protected int m_numTotalPairsTest = 0;  // the overall number of pairs in the test split

  protected int m_numPotentialDupePairsTrain = 0;
  protected int m_numActualDupePairsTrain = 0;
  protected int m_numPotentialNonDupePairsTrain = 0;
  protected int m_numActualNonDupePairsTrain = 0;

  protected double m_trainTime = 0;
  protected double m_testTimeStart = 0;
  
  /** Given training data, build the metrics required by the deduper
   * @param train a set of training data
   */
  public void buildDeduper(Instances trainFold, Instances testInstances) throws Exception {
    Instances trainInstances = getTrainingSet(trainFold);
    m_numTotalPairsTrain = trainInstances.numInstances() * (trainInstances.numInstances() - 1) / 2;
    m_numPotentialDupePairsTrain = numTruePairs(trainInstances);
    m_numPotentialNonDupePairsTrain = m_numTotalPairsTrain - m_numPotentialDupePairsTrain;

    // if the indexes have not been set, use all except for class
    if (m_attrIdxs == null) {
      m_attrIdxs = new int[trainInstances.numAttributes() - 1];
      int classIdx = trainInstances.classIndex();
      int counter = 0;
      for (int i = 0; i < m_attrIdxs.length + 1; i++) {
	if (i != classIdx) {
	  m_attrIdxs[counter++] = i;
	}
      } 
    }

    // train the instance metric
    long trainTimeStart = System.currentTimeMillis();
    m_metric.buildInstanceMetric(m_attrIdxs);
    m_metric.trainInstanceMetric(trainInstances, testInstances);

    // get training statistics
    m_numActualDupePairsTrain = m_metric.getNumActualPosPairs();   
    m_numActualNonDupePairsTrain = m_metric.getNumActualNegPairs();  
      
    m_trainTime = (System.currentTimeMillis() - trainTimeStart)/1000.0;
    m_distanceMatrix = null;
  }

  /** Identify duplicates within the testing data
   * @param testInstances a set of instances among which to identify duplicates
   * @param numObjects the number of "true object" sets to create
   * @return a list of object sets
   */
  public void findDuplicates(Instances testInstances, int numObjects) throws Exception {
    m_numObjects = testInstances.numClasses();
    m_numTruePairs = numTruePairs(testInstances);
    m_numTotalPairsTest = testInstances.numInstances() * (testInstances.numInstances() -1) / 2;
    resetStatistics();
    hashInstances(testInstances);
    createDistanceMatrix();

    // assign instances to singleton clusters
    m_numCurrentObjects = testInstances.numInstances();
    m_clusterAssignments = new int[testInstances.numInstances()];
    for (int i = 0; i < testInstances.numInstances(); i++) {
      m_clusterAssignments[i] = i;
    }

    // initialize m_clusters arraylist
    initIntClusters();
    if (m_debug) {
      System.out.println("Starting with  " + (m_numCurrentObjects) + " clusters; " + m_clusters.size() +
			 "actual clusters; " + numObjects + " true objects desired");
    }
    // merge clusters until desired number of clusters is reached
    while (m_numCurrentObjects > numObjects) {
      if (m_debug) {
	System.out.println("Merging with " + (m_numCurrentObjects) + " clusters left");
      }
      mergeStep();
    }
    System.out.println("Done deduping with " + m_clusters.size() + " clusters");
  }

    /**
   * Computes the clusters from the cluster assignments
   * 
   * @exception Exception if clusters could not be computed successfully
   */    
  public ArrayList initIntClusters() throws Exception {
    m_clusters = new ArrayList();

    for (int i=0; i < m_testInstances.numInstances(); i++) {
      m_clusters.add(new Cluster(new Integer(i)));
    }

    return m_clusters;
  }
  
  /**
   * Internal method that finds two most similar clusters and merges them
   */
  protected void mergeStep() throws Exception{
    double bestDistance = Double.MAX_VALUE;
    double thisDistance;
    Cluster thisCluster, nextCluster;
    ArrayList mergeCandidatesList = new ArrayList();
    int cluster1_index, cluster2_index;

    if (m_debug) { 
      System.out.println("\nBefore merge step there are " + m_clusters.size() +
			 " clusters; m_numCurrentObjects=" + m_numCurrentObjects);
    }
    // find two most similar clusters 
    for (int i = 0; i < m_clusters.size()-1; i++){
      thisCluster = (Cluster)m_clusters.get(i);
      for (int j = i+1; j < m_clusters.size(); j++) {
	thisDistance = clusterDistance(thisCluster, (Cluster) m_clusters.get(j));
	// If there is a tie, add to the list of top distances
	if (thisDistance == bestDistance) {
	  mergeCandidatesList.add(new Integer(i));
	  mergeCandidatesList.add(new Integer(j));
	} else if (thisDistance < bestDistance) {  // this is the best distance seen this far
	  mergeCandidatesList.clear();
	  mergeCandidatesList.add(new Integer(i));
	  mergeCandidatesList.add(new Integer(j));
	  bestDistance = thisDistance;
	}
      }
    }

    // randomly pick a most similar pair from the list of candidates
    int i1 = (int) (mergeCandidatesList.size() * Math.random());
    int i2 = (i1 % 2 > 0) ? (i1 - 1) : (i1 + 1);
    int cluster1Idx = ((Integer) mergeCandidatesList.get(i1)).intValue();
    int cluster2Idx = ((Integer) mergeCandidatesList.get(i2)).intValue();
    if (m_debug) {
      System.out.println("\nMerging clusters " + cluster1Idx + "(" + ((Cluster)m_clusters.get(cluster1Idx)).get(0) + 
			 ") and " + cluster2Idx + "(" + ((Cluster)m_clusters.get(cluster2Idx)).get(0) + 
			 ");  distance=" + bestDistance +
			 " actual=" + clusterDistance((Cluster)m_clusters.get(cluster1Idx),
						      (Cluster)m_clusters.get(cluster2Idx)));
      Instance in1 =  (Instance) m_instancesHash.get(((Cluster)m_clusters.get(cluster1Idx)).get(0));
      Instance in2 =  (Instance) m_instancesHash.get(((Cluster)m_clusters.get(cluster2Idx)).get(0));
      if (in1.classValue() == in2.classValue()) {
	System.out.println("good: " + bestDistance + "\t" + in1 + "\tand" + in2);
      } else {
	System.out.println("BAD:  " + bestDistance + "\t" + in1 + "\tand" + in2);
      } 
    }

    Cluster newCluster = mergeClusters(cluster1Idx, cluster2Idx);
    // have to remove in order because we're using index, argh
    if (cluster1Idx > cluster2Idx) {
      m_clusters.remove(cluster1Idx);
      m_clusters.remove(cluster2Idx);
    } else {
      m_clusters.remove(cluster2Idx);
      m_clusters.remove(cluster1Idx);
    }
    m_clusters.add(newCluster);
    m_numCurrentObjects--;
  }

    /**
   * internal method that returns the distance between two clusters
   */
  protected double clusterDistance(Cluster cluster1, Cluster cluster2) {
    if (cluster2 == null || cluster1 == null) {
      try{
	printIntClusters();
      } catch(Exception e){}
    }
	
    int i1 = ((Integer) cluster1.get(0)).intValue();
    int i2 = ((Integer) cluster2.get(0)).intValue();
    return m_distanceMatrix[i1][i2];
  }

     boolean fuckedUp = false;
    /** Internal method to merge two clusters and update distances
   */
  protected Cluster mergeClusters (int cluster1Idx, int cluster2Idx) throws Exception {

    Cluster newCluster = new Cluster();
    Cluster cluster1 = (Cluster) m_clusters.get(cluster1Idx);
    Cluster cluster2 = (Cluster) m_clusters.get(cluster2Idx);
    int cluster1FirstIdx =((Integer) cluster1.get(0)).intValue();
    int cluster2FirstIdx =((Integer) cluster2.get(0)).intValue(); 
    newCluster.copyElements(cluster1);
    newCluster.copyElements(cluster2);
    
    // Update the distance matrix depending on the linkage type
    // go through all clusters and update the distance from first element
    // to the first element of the new cluster
    for (int i = 0; i < m_clusters.size(); i++){
      if (i != cluster1Idx && i != cluster2Idx) { // skip these clusters themselves
	Cluster currentCluster = (Cluster) m_clusters.get(i);
	int currClusterFirstIdx = ((Integer) currentCluster.get(0)).intValue();

	if (m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] <=
	    m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx]) {
	  // first cluster is closer, no need to update
	  // but just in case:
//  	  m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx] =
//  	    m_distanceMatrix[currClusterFirstIdx][cluster2FirstIdx] =
//  	    m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx];
	} else {
	  // second cluster is closer, must update distance between the first representative
	  m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] =
	    m_distanceMatrix[currClusterFirstIdx][cluster1FirstIdx] =
	    m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx];
	}
	// check for infinity links
	if (m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx] == Double.POSITIVE_INFINITY) {
	  m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] =
	    m_distanceMatrix[currClusterFirstIdx][cluster1FirstIdx] = Double.POSITIVE_INFINITY;
	}
	if (m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] == Double.POSITIVE_INFINITY) {
	  m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx] =
	    m_distanceMatrix[currClusterFirstIdx][cluster2FirstIdx] = Double.POSITIVE_INFINITY;
	}
      }
    }
    
    // update pair statistics
    m_numTotalPairs += cluster1.size() * cluster2.size();
    int newGoodPairs = numCrossClusterTruePairs(cluster1, cluster2); 
    m_numGoodPairs += newGoodPairs;
    accumulateStatistics();

    return newCluster;
  }
  
  /**
   * Create the hashtable from given Instances;
   * keys are numeric indeces, values are actual Instances
   *
   * @param data Instances
   *
   */
  protected void hashInstances (Instances data) {
    m_testInstances = data;
    m_instancesHash = new HashMap();
    m_reverseInstancesHash = new HashMap();
    m_classValues = new double[data.numInstances()];
        
    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i);
      m_classValues[i] = instance.classValue();
      if (!m_instancesHash.containsValue(instance)) {
	Integer idx = new Integer(i);
	m_instancesHash.put(idx, instance);
	m_reverseInstancesHash.put(instance, idx);
      } else {
	System.out.println("STupid fuck, dupe! " + i);
      }
    }
  }
  

  /**
   * Fill the distance matrix with values using the metric
   *
   */
  protected void createDistanceMatrix () throws Exception {
    int n = m_instancesHash.size();
    double sim;
	
    m_distanceMatrix = new double[n][n];

    if (m_useBlocking) {
      for (int i = 0; i < n; i++) {
	Arrays.fill(m_distanceMatrix[i], Double.MAX_VALUE);
      }
      
      Blocking blocker = new Blocking();
      blocker.buildIndex(m_testInstances);
      InstancePair[] pairs = blocker.getMostSimilarPairs(m_testInstances.numClasses() * 50);
      for (int i = 0; i < pairs.length && pairs[i] != null; i++) {
	int idx1 = ((Integer) m_reverseInstancesHash.get(pairs[i].instance1)).intValue();
	int idx2 = ((Integer) m_reverseInstancesHash.get(pairs[i].instance2)).intValue();
	m_distanceMatrix[idx1][idx2] = m_distanceMatrix[idx1][idx2] = pairs[i].value;
      }
    }
    
    for (int i = 0; i < n; i++) {
      for (int j = i+1; j < n; j++) {
	if (!m_useBlocking || m_distanceMatrix[i][j] != Double.MAX_VALUE) {
	  m_distanceMatrix[i][j] = m_distanceMatrix[j][i] =
	    m_metric.distance((Instance) m_instancesHash.get(new Integer(i)),
			      (Instance) m_instancesHash.get(new Integer(j)));
	  Instance i1 = (Instance) m_instancesHash.get(new Integer(i));
	  Instance j1 = (Instance) m_instancesHash.get(new Integer(j));
	}
      }
    }
  }

    /** Outputs the current clustering
   *
   * @exception Exception if something goes wrong
   */
  public void printIntClusters() throws Exception {
    if (m_clusters == null)
      throw new Exception ("Clusters were not created");

    for (int i = 0; i < m_clusters.size(); i++) {
      Cluster cluster = (Cluster) m_clusters.get(i);
      System.out.println ("Cluster " + i + " consists of " + cluster.size() + " elements");
      for (int j = 0; j < cluster.size(); j++) {
	//		Instance instance = (Instance) m_instancesHash.get((Integer) cluster.elementAt(j));
	Integer idx = (Integer) cluster.get(j);
	Instance instance = (Instance) m_instancesHash.get(idx);
	System.out.println("\t\t" + instance);
      }
    }
  }

  /** A helper function that stratifies a training set and selects a proportion of
   * true objects for training
   * @param instances a set of instances from which to select the training data
   * @return a subset of those instances
   */
  Instances getTrainingSet(Instances instances) {
    HashMap classHash = new HashMap();
    int numTotalInstances = instances.numInstances();
    Random rand = new Random(numTotalInstances);
    Instances trainInstances = new Instances(instances, (int) (m_trainProportion * numTotalInstances));

    // hash each class 
    for (int i=0; i < instances.numInstances(); i++) {
      Instance instance = instances.instance(i);
      Double classValue = new Double(instance.classValue());
      if (classHash.containsKey(classValue)) {
	ArrayList list = (ArrayList) classHash.get(classValue);
	list.add(instance);
      } else {
	// this class has not been seen before, create an entry for it
	ArrayList list = new ArrayList();
	list.add(instance);
	classHash.put(classValue, list);
      }
    }

    // select a desired proportion of classes
    ArrayList[] classes = new ArrayList[classHash.size()];
    classes = (ArrayList[]) classHash.values().toArray(classes);
    int numClasses = classes.length;
    int[] indeces = PairwiseSelector.randomSubset((int) (m_trainProportion * numClasses), numClasses);

    for (int i = 0; i < indeces.length; i++) {
      for (int j = 0; j < classes[i].size(); j++) {
	Instance instance = (Instance) classes[i].get(j);
	trainInstances.add(instance);
      }
    } 

    return trainInstances;
  }

  /** Set the amount of training
   * @param trainProportion the proportion of the training set that will be used for learning
   */
  public void setTrainProportion(double trainProportion) {
    m_trainProportion = trainProportion;
  }

  /** Get the amount of training
   * @return the proportion of the training set that will be used for learning
   */
  public double getTrainProportion() {
    return m_trainProportion;
  }


  /** Given a test set, calculate the number of true pairs
   * @param instances a set of objects, class has the true object ID
   * @returns the number of true same-class pairs
   */
  protected int numTruePairs(Instances instances) {
    int numTruePairs = 0;
    // get the class counts
    HashMap classCountMap = new HashMap();

    for (int i = 0; i < instances.numInstances(); i++) {
      Instance instance = instances.instance(i);
      Double classValue = new Double(instance.classValue());
      if (classCountMap.containsKey(classValue)) {
	Integer counts = (Integer) classCountMap.get(classValue);
	classCountMap.put(classValue, new Integer(counts.intValue() + 1));
      } else {
	classCountMap.put(classValue, new Integer(1));
      }
    }
    
    // calculate the number of pairs
    Iterator iterator = classCountMap.values().iterator();
    while (iterator.hasNext()) {
      int counts = ((Integer) iterator.next()).intValue();
      numTruePairs += counts * (counts - 1) / 2;
    }

    return numTruePairs;
  }

  /** Given two clusters, calculate the number of true pairs that
   * will be added when the clusters are merged
   * @param cluster1 the first cluster to merge
   * @param cluster2 the second cluster to merge
   * @returns the number of true pairs that will appear once clusters are merged
   */
  protected int numCrossClusterTruePairs(Cluster cluster1, Cluster cluster2) {
    int numCCTruePairs = 0;
    int[] classCounts1 = new int[m_numObjects];
    for (int i = 0; i < cluster1.size(); i++) {
      Integer instanceIdx = (Integer) cluster1.get(i);
      classCounts1[(int)m_classValues[instanceIdx.intValue()]]++;
    }

    int[] classCounts2 = new int[m_numObjects];
    for (int i = 0; i < cluster2.size(); i++) {
      Integer instanceIdx = (Integer) cluster2.get(i);
      classCounts2[(int)m_classValues[instanceIdx.intValue()]]++;
    }

    for (int i = 0; i < m_numObjects; i++) {
      numCCTruePairs += classCounts1[i] * classCounts2[i];
      if (classCounts1[i] != 0 || classCounts2[i] != 0) { 
//  	System.out.println(i + "\t" + classCounts1[i] + "\t" + classCounts2[i]);
      }
    }
    return numCCTruePairs;
  } 

  /** Add the current state of things to statistics */
  protected void accumulateStatistics() {
    Object[] currentStats = new Object[16];

    double precision = (m_numGoodPairs+0.0)/m_numTotalPairs;
    double recall = (m_numGoodPairs+0.0)/m_numTruePairs;

    double fmeasure = 0;
    if (precision > 0) {  // avoid divide by zero in the p=0&r=0 case
      fmeasure = 2 * (precision * recall) / (precision + recall);
    }

    int statIdx = 0;
    currentStats[statIdx++] = new Double(m_numCurrentObjects);

    // Accuracy statistics
    currentStats[statIdx++] = new Double(recall);
    currentStats[statIdx++] = new Double(precision);
    currentStats[statIdx++] = new Double(fmeasure);

    // Dupe density statistics
    currentStats[statIdx++] = new Double(m_numTotalPairsTrain);
    currentStats[statIdx++] = new Double(m_numPotentialDupePairsTrain);
    currentStats[statIdx++] = new Double(m_numActualDupePairsTrain);
    currentStats[statIdx++] = new Double(m_numPotentialNonDupePairsTrain);
    currentStats[statIdx++] = new Double(m_numActualNonDupePairsTrain);
    currentStats[statIdx++] = new Double((m_numActualNonDupePairsTrain > 0) ?
					 ((m_numActualDupePairsTrain+0.0)/m_numActualNonDupePairsTrain) : 0);
    currentStats[statIdx++] = new Double((m_numPotentialDupePairsTrain+0.0)/m_numTotalPairsTrain);
    currentStats[statIdx++] = new Double(m_numTotalPairsTest);    
    currentStats[statIdx++] = new Double(m_numTruePairs);     
    currentStats[statIdx++] = new Double((m_numTruePairs + 0.0)/m_numTotalPairsTest);

    // Timing statistics
    currentStats[statIdx++] = new Double(m_trainTime);
    currentStats[statIdx++] = new Double((System.currentTimeMillis() - m_testTimeStart)/1000.0);

    m_statistics.add(currentStats);
  }

  /** Reset the current statistics */
  protected void resetStatistics() {
    m_statistics = new ArrayList();
    m_numGoodPairs = 0;
    m_numTotalPairs = 0;
    m_testTimeStart = System.currentTimeMillis();
  } 



  /** Set the InstanceMetric that is used
   * @param metric the InstanceMetric that is used to dedupe
   */
  public void setMetric(InstanceMetric metric) {
    m_metric = metric;
  }


  /** Get the InstanceMetric that is used
   * @return the InstanceMetric that is used to dedupe
   */
  public InstanceMetric getMetric() {
    return m_metric;
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

  /** Turn debugging output on/off
   * @param debug if true, blocking is on
   */
  public void setUseBlocking(boolean useBlocking) {
    m_useBlocking = useBlocking;
  }

  /** See whether blocking is on/off
   * @returns if true, blocking is on
   */
  public boolean getUseBlocking() {
    return m_useBlocking;
  }

  
    /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   **/
  public Enumeration listOptions() {
    Vector newVector = new Vector(2);
    newVector.addElement(new Option("\tMetric.\n"
				    +"\t(default=ClassifierInstanceMetric)", "M", 1,"-M metric_name metric_options"));
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   *
   * Valid options are:<p>
   *
   * -M metric options <p>
   * InstanceMetric used <p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   *
   **/
  public void setOptions(String[] options) throws Exception {
    String optionString;

    String metricString = Utils.getOption('M', options);
    if (metricString.length() != 0) {
      String[] metricSpec = Utils.splitOptions(metricString);
      String metricName = metricSpec[0]; 
      metricSpec[0] = "";
      System.out.println("Metric name: " + metricName + "\nMetric parameters: " + concatStringArray(metricSpec));
      setMetric(InstanceMetric.forName(metricName, metricSpec));
    }
  }


  /**
   * Gets the current settings of Greedy Agglomerative Clustering
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    String [] options = new String [250];
    int current = 0;

    if (m_useBlocking == false) { 
      options[current++] = "-NB"; 
    }

    options[current++] = "-T";
    options[current++] = "" + m_trainProportion;

    if (m_debug) {
      options[current++] = "-D";
    }

    options[current++] = "-M";
    options[current++] = Utils.removeSubstring(m_metric.getClass().getName(), "weka.deduping.metrics.");
    if (m_metric instanceof OptionHandler) {
      String[] metricOptions = ((OptionHandler)m_metric).getOptions();
      for (int i = 0; i < metricOptions.length; i++) {
	options[current++] = metricOptions[i];
      }
    } 

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }

  /** A little helper to create a single String from an array of Strings
   * @param strings an array of strings
   * @returns a single concatenated string
   */
  public static String concatStringArray(String[] strings) {
    StringBuffer buffer = new StringBuffer();
    for (int i = 0; i < strings.length; i++) {
      buffer.append(strings[i]);
      buffer.append(" ");
    }
    return buffer.toString();
  } 
}







