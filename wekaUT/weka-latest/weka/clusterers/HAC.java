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
 *    MatrixHAC.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */


/**
 * Similarity-matrix Implementation of Hierarachical Agglomerative Clustering.
 * <p>
 * Valid options are:<p>
 *
 * -N <0-10000> <br>
 * Number of clusters. <p>
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.11 $
 */
package weka.clusterers;

import java.io.*;
import java.util.*;
import java.text.*;

import weka.core.*;
import weka.core.metrics.*;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.Filter;

public class HAC extends Clusterer implements SemiSupClusterer, OptionHandler{
  /* name of the clusterer */
  String m_name = "HAC";

  /** Number of clusters */
  protected int m_numClusters = -1; 

  /** Number of clusters in the process*/
  protected int m_numCurrentClusters = 0; 

  /** ID of current cluster */
  protected int m_clusterID = 0;

  /** Number of seeded clusters */
  protected int m_numSeededClusters = 0;

  /** Dot file name for dumping graph for tree visualization */
  protected String m_dotFileName = "user-features.dot";

  /** Dot file name for dumping graph for tree visualization */
  protected PrintWriter m_dotWriter = null;
      
  /** Instances that we are working with */
  Instances m_instances;
  Instances m_descrInstances; 

  /** holds the clusters */
  protected ArrayList m_clusters = null;

  /**
   * temporary variable holding cluster assignments
   */
  protected int [] m_clusterAssignments;
  
  /** distance matrix */
    protected double[][] m_distanceMatrix = null;

  /** cluster similarity type */
  public final static int SINGLE_LINK = 0;
  public final static int COMPLETE_LINK = 1;
  public final static int GROUP_AVERAGE = 2;
  public static final Tag[] TAGS_LINKING = {
    new Tag(SINGLE_LINK, "Single link"),
    new Tag(COMPLETE_LINK, "Complete link"),
    new Tag(GROUP_AVERAGE, "Group-average")
      };

  /** Default linking method */
  protected int m_linkingType = GROUP_AVERAGE; 
    
  /** starting index of test data in unlabeledData if transductive clustering */
  protected int m_StartingIndexOfTest = -1;
  
  /** seeding  */
  protected boolean m_seedable = false;
  /** holds the ([seed instance] -> [clusterLabel of seed instance]) mapping */
  protected HashMap m_SeedHash = null;


  /** A 'checksum hash' where indices are hashed to the sum of their attribute values */
  protected HashMap m_checksumHash = null;
  protected double[] m_checksumPerturb = null;
  
  /**
   * holds the random Seed, useful for random selection initialization
   */
  protected int m_randomSeed = 100;
  protected Random m_randomGen = null;

  /** instance hash */
  protected HashMap m_instancesHash = null;

  /** reverse instance hash */
  protected HashMap m_reverseInstancesHash = null;

  /** The threshold distance beyond which no clusters are merged (except for one - TODO) */
  protected double m_mergeThreshold = 0.8; 

  /** verbose? */
  protected boolean m_verbose = false;

  /** metric used to calculate similarity/distance */
//    protected Metric m_metric = new WeightedDotP();
//    protected String m_metricName = new String("weka.core.metrics.WeightedDotP");
  protected Metric m_metric = new WeightedEuclidean();
    protected String m_metricName = new String("weka.core.metrics.WeightedEuclidean");
  /** Is the metric (and hence the algorithm) relying on similarities or distances? */
  protected boolean  m_isDistanceBased = false;

  /** has the metric has been constructed?  a fix for multiple buildClusterer's */
  protected boolean m_metricBuilt = false;
  
  // ===============
  // Public methods.
  // ===============

  /** empty constructor, required to call using Class.forName */
  public HAC() {}


  /* Constructor */
  public HAC(Metric metric) {
    m_metric = metric;
    m_metricName = m_metric.getClass().getName();
    m_isDistanceBased = metric.isDistanceBased();
  }

  /** Sets training instances */
  public void setInstances(Instances instances) {
    m_instances = instances;
  }


  /** Return training instances */
  public Instances getInstances() {
    return m_instances;
  }

  /**
   * Set the number of clusters to generate
   *
   * @param n the number of clusters to generate
   */
  public void setNumClusters(int n) {
    m_numClusters = n;
    if (m_verbose) {
      System.out.println("Number of clusters: " + n);
    }
  }

  /** Set the merge threshold */
  public void setMergeThreshold(double threshold) {
    m_mergeThreshold = threshold;
  }

  /** Get the merge threshold */
  public double getMergeThreshold() {
    return m_mergeThreshold;
  }


  /**
   * Set the distance metric
   *
   * @param s the metric
   */
  public void setMetric (LearnableMetric m) {
    m_metric = m;
    m_metricName = m_metric.getClass().getName();
    m_isDistanceBased = m.isDistanceBased();
  }

  /**
   * Get the distance metric
   *
   * @returns the distance metric used
   */
  public Metric getMetric () {
    return m_metric;
  }

  /**
   * Get the distance metric name
   *
   * @returns the name of the distance metric used
   */
  public String metricName () {
    return m_metricName;
  }
  
  /**
   * We always want to implement SemiSupClusterer from a class extending Clusterer.  
   * We want to be able to return the underlying parent class.
   * @return parent Clusterer class
   */
  public Clusterer getThisClusterer() {
    return this;
  }

  /**
   * Cluster given instances to form the specified number of clusters.
   *
   * @param data instances to be clustered
   * @param num_clusters number of clusters to create
   * @exception Exception if something goes wrong.
   */
  public void buildClusterer(Instances data, int num_clusters) throws Exception {
    setNumClusters(num_clusters);
    buildClusterer(data);
  }

  /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using labeledData as seeds
   *
   * @param labeledData labeled instances to be used as seeds
   * @param unlabeledData unlabeled instances
   * @param classIndex attribute index in labeledData which holds class info
   * @param numClusters number of clusters
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(Instances labeledData, Instances unlabeledData,
			     int classIndex, int numClusters) throws Exception {
    // remove labels of labeledData before putting in SeedHash
    Instances clusterData = new Instances(labeledData);
    clusterData.deleteClassAttribute();

    // create SeedHash from labeledData
    if (getSeedable()) {
      Seeder seeder = new Seeder(clusterData, labeledData);
      setSeedHash(seeder.getAllSeeds());
    }

    // add unlabeled data to labeled data (labels removed), not the
    // other way around, so that the hash table entries are consistent
    // with the labeled data without labels
    for (int i=0; i<unlabeledData.numInstances(); i++) {
      clusterData.add(unlabeledData.instance(i));
    }
    
    if (m_verbose) {
      System.out.println("combinedData has size: " + clusterData.numInstances() + "\n");
    }

    // learn metric using labeled data, then  cluster both the labeled and unlabeled data
    m_metric.buildMetric(labeledData);
    m_metricBuilt = true;

    // check if the number of clusters is dynamically set to the number of classes
    if (m_numClusters == -1) {
      m_numClusters = labeledData.numClasses();
      System.out.println("DYNAMIC NUMBER OF CLUSTERS, setting to " + m_numClusters);
    }
    buildClusterer(clusterData, numClusters);
  }

    /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using labeledData as seeds
   *
   * @param labeledData labeled instances to be used as seeds
   * @param unlabeledData unlabeled instances
   * @param classIndex attribute index in labeledData which holds class info
   * @param numClusters number of clusters
   * @param startingIndexOfTest from where test data starts in unlabeledData, useful if clustering is transductive
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(Instances labeledData, Instances unlabeledData, int classIndex,
			     int numClusters, int startingIndexOfTest) throws Exception {
    m_StartingIndexOfTest = startingIndexOfTest + labeledData.numInstances();
    buildClusterer(labeledData, unlabeledData, classIndex, numClusters);
  }


  /**
   * Cluster given instances.  If no threshold or number of clusters is set,
   * clustering proceeds until two clusters are left. 
   *
   * @param data instances to be clustered
   * @exception Exception if something goes wrong.
   */
  public void buildClusterer(Instances data) throws Exception {
    m_randomGen = new Random(m_randomSeed);
    m_dotWriter = new PrintWriter(new BufferedOutputStream(new FileOutputStream(m_dotFileName)));
    m_dotWriter.println("digraph HAC {\n");

    setInstances(data);
    m_clusterAssignments = new int[m_instances.numInstances()];
    
    if (m_verbose && m_SeedHash != null) {
      System.out.println("Using seeding ...");
    }

    if (m_instances.checkForNominalAttributes() && m_instances.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle nominal attributes\n");
    }

    //m_instances = filterInstanceDescriptions(m_instances);
    // Don't rebuild the metric if it was already trained
    if (!m_metricBuilt) {
      m_metric.buildMetric(data);
    }
    
    hashInstances(m_instances);
    createDistanceMatrix();
    cluster();
    unhashClusters();

    m_dotWriter.println("}");
    m_dotWriter.close();
  }

  /** If some of the attributes start with "__", form a separate Instances set with
   * descriptions and filter them out of the argument dataset.
   * Return the original dataset without the filtered out attributes
   */
  protected Instances filterInstanceDescriptions(Instances instances) throws Exception {
    Instances filteredInstances;

//      Normalize normalizeFilter = new Normalize();
//      normalizeFilter.setInputFormat(instances);
//      instances = Filter.useFilter(instances, normalizeFilter);
//      System.out.println("Normalized the instance attributes");
    
    // go through the attributes and find the description attributes
    ArrayList descrIndexList = new ArrayList();
    for (int i = 0; i < instances.numAttributes(); i++) {
      Attribute attr = instances.attribute(i);
      if (attr.name().startsWith("__")) {
	descrIndexList.add(new Integer(i));
	System.out.println("filtering " + attr);
      } 
    }
 

    // filter out the description attributes if necessary
    if (descrIndexList.size() > 0) {
      m_descrInstances = new Instances(instances);

      // filter out the descriptions first
      int[] descrIndeces = new int[descrIndexList.size()];
      for (int i = 0; i < descrIndexList.size(); i++) {
	descrIndeces[i] = ((Integer) descrIndexList.get(i)).intValue();
      }

      Remove attributeFilter = new Remove();
      attributeFilter.setAttributeIndicesArray(descrIndeces);
      attributeFilter.setInvertSelection(false);
      attributeFilter.setInputFormat(instances);
      filteredInstances = Filter.useFilter(instances, attributeFilter);

      attributeFilter.setInvertSelection(true);
      attributeFilter.setInputFormat(instances);
      m_descrInstances = Filter.useFilter(instances, attributeFilter);
    } else {
      filteredInstances = new Instances(instances);
    }
    return filteredInstances;
  } 
  
 
  /**
   * Reset all values that have been learned
   */
  public void resetClusterer()  throws Exception{
    if (m_metric instanceof LearnableMetric)
      ((LearnableMetric)m_metric).resetMetric();
    m_SeedHash = null;
  }

  /** Set the m_SeedHash */
  public void setSeedHash(HashMap seedhash) {
    m_SeedHash = seedhash;
    m_seedable = true;
  }    

 /**
   * Set the random number seed
   * @param s the seed
   */
  public void setRandomSeed (int s) {
    m_randomSeed = s;
  }

    
  /** Return the random number seed */
  public int getRandomSeed () {
    return  m_randomSeed;
  }
  

  /** Turn seeding on and off
   * @param seedable should seeding be done?
   */
  public void setSeedable(boolean seedable) {
    m_seedable = seedable;
  }


  /** Turn seeding on and off
   * @param seedable should seeding be done?
   */
  public boolean getSeedable() {
    return m_seedable;
  }

     
  /** Read the seeds from a hastable, where every key is an instance and every value is:
   * a FastVector of Doubles: [(Double) probInCluster0 ... (Double) probInClusterN]
   * @param seedVector vector containing seeds
   */
  public void seedClusterer(HashMap SeedHash) {
    if (m_seedable) {
      m_SeedHash = SeedHash;
    }
  }

  /**
   * returns the SeedHash
   * @return seeds hash
   */
  public HashMap getSeedHash() {return m_SeedHash;}

  /**
   * Create the hashtable from given Instances;
   * keys are numeric indeces, values are actual Instances
   *
   * @param data Instances
   *
   */
  protected void hashInstances (Instances data) {
    int next_value = 0;

    m_instancesHash = new HashMap();
    m_reverseInstancesHash = new HashMap();
    m_checksumHash = new HashMap();
    // initialize checksum perturbations
    m_checksumPerturb = new double[data.numAttributes()];
    for (int i = 0; i < m_checksumPerturb.length; i++) {
      m_checksumPerturb[i] = m_randomGen.nextFloat();
    }
    
    for (Enumeration enum = data.enumerateInstances(); enum.hasMoreElements();) {
      Instance instance = (Instance) enum.nextElement();
      if (!m_instancesHash.containsValue(instance)) {
	Integer idx = new Integer(next_value);
	next_value++;
	m_instancesHash.put(idx, instance);
	m_reverseInstancesHash.put(instance, idx);
      
	// hash the checksum value
	double [] values = instance.toDoubleArray();
	double checksum = 0; 
	for (int i = 0; i < values.length; i++) { checksum += m_checksumPerturb[i] * values[i];}
	Double checksumIdx = new Double(checksum);
	if (m_checksumHash.containsKey(checksumIdx)) {
	  Object prev = m_checksumHash.get(checksumIdx);
	  ArrayList chain; 
	  if (prev instanceof Integer) { 
	    chain = new ArrayList();
	    Integer prevIdx = (Integer) m_checksumHash.get(checksumIdx);
	    chain.add(prevIdx);
	  } else {  //instanceof Arraylist
	    chain = (ArrayList) m_checksumHash.get(checksumIdx);
	  }
	  chain.add(idx);
	  m_checksumHash.put(checksumIdx, chain);
	} else {  // no collisions
	  m_checksumHash.put(checksumIdx, idx);
	}
      } else {
	System.err.println("Already encountered instance, skipping " + instance);
      }
    }
  }

  /**
   * assuming m_clusters contains the clusters of indeces, convert it to
   * clusters containing actual instances
   */ 
  protected void unhashClusters() throws Exception{
    if (m_clusters == null  || m_instancesHash == null)
      throw new Exception ("Clusters or hash not initialized");

    ArrayList clusters = new ArrayList();
    for (int i = 0; i < m_clusters.size(); i++ ) {
      Cluster cluster = (Cluster) m_clusters.get(i);
      Cluster newCluster = new Cluster();
      for (int j = 0; j < cluster.size(); j++) {
	Integer instanceIdx = (Integer) cluster.get(j);
	double wt = cluster.weightAt(j);
	newCluster.add((Instance)m_instancesHash.get(instanceIdx), wt);
      }
      clusters.add(newCluster);
    }
    m_clusters = clusters;
  }
	

  /**
   * Fill the distance matrix with values using the metric
   *
   */
  protected void createDistanceMatrix () throws Exception {
    int n = m_instancesHash.size();
    double sim;
	
    m_distanceMatrix = new double[n][n];

    for (int i = 0; i < n; i++) {
      for (int j = i+1; j < n; j++) {
	m_distanceMatrix[i][j] = m_distanceMatrix[j][i] =
	  m_metric.distance((Instance) m_instancesHash.get(new Integer(i)),
			    (Instance) m_instancesHash.get(new Integer(j)));
      }
    }
  }

  /**
   * Set the type of clustering
   *
   * @param type Clustering type: can be HAC.SINGLE_LINK, HAC.COMPLETE_LINK,
   * or HAC.GROUP_AVERAGE
   */
  public void setLinkingType (SelectedTag linkingType)
  {
    if (linkingType.getTags() == TAGS_LINKING) {
      m_linkingType = linkingType.getSelectedTag().getID();
    }
  }

  /**
   * Get the linking type
   *
   * @returns the linking type
   */
  public SelectedTag getLinkingType ()
  {
      return new SelectedTag(m_linkingType, TAGS_LINKING);
  }


  /**
   * Internal method that initializes distances between seed clusters to
   * POSITIVE_INFINITY
   */
  protected void initConstraints() {
    for (int i = 0; i < m_instances.numInstances(); i++) {
      if (m_clusterAssignments[i] < m_numSeededClusters) {
	// make distances to elements from other seeded clusters POSITIVE_INFINITY
	for (int j = i+1; j < m_instances.numInstances(); j++) {
	  if (m_clusterAssignments[j] < m_numSeededClusters &&
	      m_clusterAssignments[j] != m_clusterAssignments[i]) {
	    m_distanceMatrix[i][j] =
	      m_distanceMatrix[j][i] = Double.POSITIVE_INFINITY;
	  }
	}
      }
    } 
  }

  /**
   * Internal method that produces the actual clusters
   */
  protected void cluster() throws Exception {
    double last_distance = Double.MIN_VALUE;
    m_numCurrentClusters = 0;
    m_numSeededClusters = 0;
    m_clusters = new ArrayList();
    TreeSet leftOverSet = null;

    // Initialize singleton clusters
    m_clusterAssignments = new int[m_instances.numInstances()];
    for (int i = 0; i < m_instances.numInstances(); i++) {
      m_clusterAssignments[i] = -1;
    }

    // utilize seeds if available
    if (m_SeedHash != null) {
      if (m_verbose) {
	System.out.println("Seeding HAC using " + m_SeedHash.size() + " seeds");
      }
      Iterator iterator = m_SeedHash.entrySet().iterator();
      int maxClassIdx = -1;
      HashSet classIdxSet = new HashSet();
      while (iterator.hasNext()) {
	Map.Entry entry = (Map.Entry) iterator.next();
	Instance instance = (Instance) entry.getKey();
	Integer instanceIdx = (Integer) m_reverseInstancesHash.get(instance);
	Integer clusterIdx = (Integer) entry.getValue();
	classIdxSet.add(clusterIdx);
	m_clusterAssignments[instanceIdx.intValue()] = clusterIdx.intValue();
	if (clusterIdx.intValue() > maxClassIdx) {
	  maxClassIdx = clusterIdx.intValue();
	}
      }
      m_numCurrentClusters = m_numSeededClusters = classIdxSet.size();
      System.out.println("Seeded " + m_numSeededClusters + " clusters");

      // If the seeding is incomplete, need to memorize "unseeded" cluster numbers
      if (m_numCurrentClusters < m_numClusters) {
	leftOverSet = new TreeSet();
	for (int i = 0; i < m_numClusters; i++) {
	  if (!classIdxSet.contains(new Integer(i))) {
	    leftOverSet.add(new Integer(i));
	  } 
	} 
      }
    }
    
    // assign unseeded instances to singleton clusters
    for (int i = 0; i < m_instances.numInstances(); i++) {
      if (m_clusterAssignments[i] == -1) {
	// utilize "left over clusters first"
	if (leftOverSet != null) {
	  Integer clusterIdx = (Integer) leftOverSet.first();
	  m_clusterAssignments[i] = clusterIdx.intValue();
	  leftOverSet.remove(clusterIdx);
	  if (leftOverSet.isEmpty()) {
	    leftOverSet = null;
	  }
	} else {
	  m_clusterAssignments[i] = m_numCurrentClusters;
	}
	m_numCurrentClusters++; 
      }
    }
    // initialize m_clusters arraylist
    getIntClusters();

    if (m_SeedHash != null) {
      initConstraints();
    }

    
    // merge clusters until desired number of clusters is reached
    double mergeDistance = 0; 
    while (m_numCurrentClusters >  m_numClusters && mergeDistance < m_mergeThreshold) {
      mergeDistance = mergeStep();
      if (m_verbose) {
	System.out.println("Merged with " + (m_numCurrentClusters) + " clusters left; distance=" + mergeDistance);
      }
    }
    System.out.println("Done clustering with " + m_clusters.size() + " clusters");
    for (int i = 0; i < m_clusters.size(); i++) System.out.print(((Cluster)m_clusters.get(i)).size() + "\t");
    initClusterAssignments();
  }

  /**
   * Internal method that finds two most similar clusters and merges them
   */
  protected double mergeStep() throws Exception{
    double bestDistance = Double.MAX_VALUE;
    double thisDistance;
    Cluster thisCluster, nextCluster;
    ArrayList mergeCandidatesList = new ArrayList();
    int cluster1_index, cluster2_index;

    if (m_verbose) { 
      System.out.println("\nBefore merge step there are " + m_clusters.size() +
			 " clusters; m_numCurrentClusters=" + m_numCurrentClusters);
    }
    // find two most similar clusters 
    for (int i = 0; i < m_clusters.size()-1; i++){
      thisCluster = (Cluster)m_clusters.get(i);
      for (int j = i+1; j < m_clusters.size(); j++) {
	thisDistance = clusterDistance(thisCluster, (Cluster) m_clusters.get(j));
	if (m_verbose) {
	  //	  System.out.println("Distance between " + i + " and " + j + " is " + thisDistance);
	}
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
    int i1 = (int) (mergeCandidatesList.size() * m_randomGen.nextFloat());
    int i2 = (i1 % 2 > 0) ? (i1 - 1) : (i1 + 1);
    int cluster1Idx = ((Integer) mergeCandidatesList.get(i1)).intValue();
    int cluster2Idx = ((Integer) mergeCandidatesList.get(i2)).intValue();

    if (m_verbose) {
      System.out.println("\nMerging clusters " + cluster1Idx + " and " + cluster2Idx + ";  distance=" + bestDistance);
    }
    System.out.print("Best distance=" + ((float)bestDistance) + "; Merging:\n");
    printCluster(cluster1Idx);
    System.out.print("AND\n");
    printCluster(cluster2Idx);
    System.out.print("\n");

    Cluster newCluster = mergeClusters(cluster1Idx, cluster2Idx);

    // check if the new cluster is sufficiently large and "good"
    HashMap groupCountMap = new HashMap(); 
    for (int i = 0; i < newCluster.size(); i++) {
      int idx = ((Integer)newCluster.get(i)).intValue();

      Instance instance = m_descrInstances.instance(idx);

      // get the set of groups
      String groupString = instance.stringValue(1);
      StringTokenizer tokenizer = new StringTokenizer(groupString, "|");
      while (tokenizer.hasMoreTokens()) {
	String group = tokenizer.nextToken();
	if (groupCountMap.containsKey(group)) {
	  Integer count = (Integer) groupCountMap.get(group);
	  groupCountMap.put(group, new Integer(count.intValue() + 1));
	} else {
	  groupCountMap.put(group, new Integer(1));
	}
      }
      
    }
    int largestGroupCount = -1;
    Iterator iterator = groupCountMap.entrySet().iterator();
    while(iterator.hasNext()) {
      Map.Entry entry = (Map.Entry) iterator.next();
      int thisCount = ((Integer)entry.getValue()).intValue();
      String group = (String) entry.getKey();
      if (thisCount > largestGroupCount && !group.equals("grad")) {
	largestGroupCount = thisCount;

      } 
    } 
    // if the most common group includes 80% of cluster members, yell!
    if ((largestGroupCount + 0.0)/(newCluster.size() + 0.0) > 0.6 && newCluster.size() > 2) {
      System.out.println("HAPPY JOY JOY!  LOOK HERE!"); 
    } 
    
    
    // have to remove in order because we're using index, argh
    if (cluster1Idx > cluster2Idx) {
      m_clusters.remove(cluster1Idx);
      m_clusters.remove(cluster2Idx);
    } else {
      m_clusters.remove(cluster2Idx);
      m_clusters.remove(cluster1Idx);
    }
    m_clusters.add(newCluster);
    m_numCurrentClusters--;
    return bestDistance;
  }
    

  /**
   * Computes the clusters from the cluster assignments
   * 
   * @exception Exception if clusters could not be computed successfully
   */    
  public ArrayList getIntClusters() throws Exception {
    m_clusters = new ArrayList();
    Cluster [] clusterArray = new Cluster[m_numCurrentClusters];

    if (m_verbose) {
      System.out.println("Cluster assignments: ");
      for (int i=0; i < m_clusterAssignments.length; i++) {
	System.out.print(i + ":" + m_clusterAssignments[i] + "  ");
      }
    }

    for (int i=0; i < m_instances.numInstances(); i++) {
	Instance inst = m_instances.instance(i);
	if(clusterArray[m_clusterAssignments[i]] == null)
	   clusterArray[m_clusterAssignments[i]] = new Cluster(m_clusterID++);
	clusterArray[m_clusterAssignments[i]].add(new Integer(i), 1);
	//	System.out.println("Adding: " + i + " to cluster: " + m_clusterAssignments[i]); 
    }

    for (int j =0; j< m_numCurrentClusters; j++) {
      if (clusterArray[j] == null) {
	System.out.println("Empty cluster: " + j);
	//	printIntClusters();
	setVerbose(true);
	m_numCurrentClusters--;
	m_numClusters--;
      } else {
	m_clusters.add(clusterArray[j]);
	String labelString = "";
	for (int i = 0; i < clusterArray[j].size(); i++){    
	  Instance inst = m_instances.instance(((Integer) (clusterArray[j].get(i))).intValue());
	  labelString = labelString + printInstance(inst) + "\\n";
	}
	m_dotWriter.println("node" + clusterArray[j].clusterID + "[label = \"" + labelString + "\"]");
      }
    }

    //    printIntClusters();
    
    return m_clusters;
  }

  /**
   * Computes the final clusters from the cluster assignments, for external access
   * 
   * @exception Exception if clusters could not be computed successfully
   */    

  public ArrayList getClusters() throws Exception {
    ArrayList finalClusters = new ArrayList();
    Cluster [] clusterArray = new Cluster[m_numClusters];

    for (int i=0; i < m_instances.numInstances(); i++) {
	Instance inst = m_instances.instance(i);
	if(clusterArray[m_clusterAssignments[i]] == null)
	   clusterArray[m_clusterAssignments[i]] = new Cluster();
	clusterArray[m_clusterAssignments[i]].add(inst, 1);
    }

    for (int j =0; j< m_numClusters; j++) 
      finalClusters.add(clusterArray[j]);

    return finalClusters;
  }

  /**
   * internal method that returns the distance between two clusters
   */
  protected double clusterDistance(Cluster cluster1, Cluster cluster2) {
    if (cluster2 == null || cluster1 == null) {
      System.err.println("PANIC!  clusterDistance called with null argument(s)");
      try{
	printIntClusters();
      } catch(Exception e){}
    }
	
    int i1 = ((Integer) cluster1.get(0)).intValue();
    int i2 = ((Integer) cluster2.get(0)).intValue();
    return m_distanceMatrix[i1][i2];
  }

  protected void checkClusters() {
  }
  

  /** Internal method to merge two clusters and update distances
   */
  protected Cluster mergeClusters (int cluster1Idx, int cluster2Idx) throws Exception {
    Cluster newCluster = new Cluster(m_clusterID++);
    Cluster cluster1 = (Cluster) m_clusters.get(cluster1Idx);
    Cluster cluster2 = (Cluster) m_clusters.get(cluster2Idx);
    int cluster1FirstIdx =((Integer) cluster1.get(0)).intValue();
    int cluster2FirstIdx =((Integer) cluster2.get(0)).intValue(); 
    newCluster.copyElements(cluster1);
    newCluster.copyElements(cluster2);
    checkClusters();
    
    // Update the distance matrix depending on the linkage type
    switch (m_linkingType) {
    case SINGLE_LINK:
      // go through all clusters and update the distance from first element
      // to the first element of the new cluster
      for (int i = 0; i < m_clusters.size(); i++){
	if (i != cluster1Idx && i != cluster2Idx) { // skip these clusters themselves
	  Cluster currentCluster = (Cluster) m_clusters.get(i);
	  int currClusterFirstIdx = ((Integer) currentCluster.get(0)).intValue();
	  if (m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] <
	      m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx]) {
	    // first cluster is closer, no need to update
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
      break;
    case COMPLETE_LINK:
      // go through all clusters and update the distance from first element
      // to the first element of the new cluster
      for (int i = 0; i < m_clusters.size(); i++){
	if (i != cluster1Idx && i != cluster2Idx) { // skip these clusters themselves
	  Cluster currentCluster = (Cluster) m_clusters.get(i);
	  int currClusterFirstIdx = ((Integer) currentCluster.get(0)).intValue();
	  if (m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] >
	      m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx]) {
	    // first cluster is closer, no need to update
	  } else {
	    // second cluster is closer, must update distance between the first representative
	    m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] =
	      m_distanceMatrix[currClusterFirstIdx][cluster1FirstIdx] =
	      m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx];
	  }
	}
      }
      break;
    case GROUP_AVERAGE:
      // go through all clusters and update the distance from first element
      // to the first element of the new cluster
      for (int i = 0; i < m_clusters.size(); i++){
	if (i != cluster1Idx && i != cluster2Idx) { // skip these clusters themselves
	  Cluster currentCluster = (Cluster) m_clusters.get(i);
	  int currClusterFirstIdx = ((Integer) currentCluster.get(0)).intValue();
	  int cluster1Size = cluster1.size();
	  int cluster2Size = cluster2.size();
	  // must update distance between the first representative
	    m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] =
	      m_distanceMatrix[currClusterFirstIdx][cluster1FirstIdx] =
	      (m_distanceMatrix[cluster1FirstIdx][currClusterFirstIdx] * cluster1Size +
	       m_distanceMatrix[cluster2FirstIdx][currClusterFirstIdx] * cluster2Size) /
	      (cluster1Size + cluster2Size);
	}
      }
    } 
    String labelString = "";
    for (int i = 0; i < newCluster.size(); i++){    
      Instance inst = m_instances.instance(((Integer) (newCluster.get(i))).intValue());
      labelString = labelString + printInstance(inst) + "\\n";
    }

    m_dotWriter.println("node" + newCluster.clusterID + "[label = \"" + labelString + "\"]");
    m_dotWriter.println("node" + newCluster.clusterID + "->node" + cluster1.clusterID);
    m_dotWriter.println("node" + newCluster.clusterID + "->node" + cluster2.clusterID);

    return newCluster;
  }

  /** Print an instance for the dot file */
  String printInstance(Instance instance) {
    String stringToPrint;
    int[] ascendingSortIndicesOfAttributes = Utils.sort(instance.toDoubleArray());
    if (m_descrInstances == null) { 
      stringToPrint = instance.toString();
    } else {
      int idx = ((Integer) m_reverseInstancesHash.get(instance)).intValue();
      stringToPrint = (m_descrInstances.instance(idx)).toString() + ":  ";
    }

    DecimalFormat fmt = new DecimalFormat("0.000");
    for (int i = 0; i < 5; i++) { 
      Attribute attrib = m_instances.attribute(ascendingSortIndicesOfAttributes[m_instances.numAttributes()-i-1]);
      if (instance.value(attrib) > 0) { 
	stringToPrint = stringToPrint + attrib.name() + ": " + fmt.format(instance.value(attrib)) + "\t";
      }
    }

    return stringToPrint;
  }

  /** Update the clusterAssignments for all points in two clusters that are about to be merged
   */
  protected void initClusterAssignments() {
    m_clusterAssignments = new int[m_instances.numInstances()];
    
    for (int i = 0; i < m_clusters.size(); i++) {
      Cluster cluster = (Cluster) m_clusters.get(i);
      for (int j = 0; j < cluster.size(); j++) {
	Integer idx = (Integer) cluster.get(j);
	//	System.out.println("Instance number: " + idx + " has cluster id: " + i);
	m_clusterAssignments[idx.intValue()] = i;
      }
    }
  }
    
  /** Outputs the current clustering
   *
   * @exception Exception if something goes wrong
   */
  public void printClusters() throws Exception {
    if (m_clusters == null)
      throw new Exception ("Clusters were not created");

    for (int i = 0; i < m_clusters.size(); i++) {
      System.out.println ("Cluster " + i);
      printCluster(i);
    }
  }

  /** Outputs the specified cluster
   *
   * @exception Exception if something goes wrong
   */
  public void printCluster(int i) throws Exception {
    if (m_clusters == null)
      throw new Exception ("Clusters were not created");

    Cluster cluster = (Cluster) m_clusters.get(i);
    for (int j = 0; j < cluster.size(); j++) {
      //		Instance instance = (Instance) m_instancesHash.get((Integer) cluster.elementAt(j));
      Object o = cluster.get(j);
      Instance instance = (o instanceof Instance) ? (Instance)o : m_instances.instance(((Integer)o).intValue());

      if (m_descrInstances == null) { 
	System.out.print("\t" + instance);
      } else {
	System.out.print("\t");
	System.out.println(printInstance(instance));
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

  /**
   * Clusters an instance.
   *
   * @param instance the instance to cluster.
   * @exception Exception if something goes wrong.
   */
  public int clusterInstance(Instance instance) throws Exception {
    double bestDistance = Double.MAX_VALUE;
    int instanceIdx = 0;

    //  if (m_reverseInstancesHash.containsKey(instance)) {
    //        instanceIdx = ((Integer) m_reverseInstancesHash.get(instance)).intValue();
    //        System.out.println("Located index in m_reverseInstancesHash");
    //        return m_clusterAssignments[instanceIdx];
    //      } else {
    
    double [] values = instance.toDoubleArray();
    double checksum = 0; 
    for (int i = 0; i < values.length; i++) {
      checksum += m_checksumPerturb[i] * values[i];
    }
    Double checksumIdx = new Double(checksum);

    if (m_checksumHash.containsKey(checksumIdx)) {
      Object obj = m_checksumHash.get(checksumIdx);
      if (obj instanceof Integer) {
	int idx = ((Integer) obj).intValue();
	return m_clusterAssignments[idx];
      } else { // instanceof Arraylist
	ArrayList chain = (ArrayList) obj;
	for (int i = 0; i < chain.size(); i++) {
	  Integer idx = (Integer) chain.get(i);
	  Instance clusteredInstance = (Instance) m_instancesHash.get(idx);
	  if (matchInstance(instance, clusteredInstance)) {
	    return m_clusterAssignments[idx.intValue()];
	  }
	}
	throw new Exception("UNKNOWN INSTANCE!!!!");
      }
	
    } else {   // unknown checksum
	throw new Exception("UNKNOWN CHECKSUM!!!!");
      

     //   ArrayList candidateClusterList = new ArrayList();

//        for (int i = 0; i < m_numClusters; i++) {
//  	Cluster thisCluster = (Cluster) m_clusters.get(i);
//  	double thisDistance = distance (instance, thisCluster);
//  	if (thisDistance < bestDistance) {
//  	  candidateClusterList.clear();
//  	  candidateClusterList.add (new Integer(i));
//  	  bestDistance = thisDistance;
//  	} else if (thisDistance == bestDistance) {
//  	  candidateClusterList.add (new Integer(i));
//  	}
//        }
//        // randomly pick a candidate
//        int i = (int) (candidateClusterList.size() * Math.random());
//        int clusterIdx = ((Integer) candidateClusterList.get(i)).intValue();
//        if (clusterIdx != m_clusterAssignments[instanceIdx]) {
//  	System.out.println("Mismatch: idx=" + clusterIdx + " assigned=" + m_clusterAssignments[instanceIdx]);
//        }
//        return clusterIdx;
    }
  }

  
  /** Internal method:  check if two instances match on their attribute values
   */
  protected boolean matchInstance(Instance instance1, Instance instance2) {
    double [] values1 = instance1.toDoubleArray();
    double [] values2 = instance2.toDoubleArray();
    for (int i = 0; i < values1.length; i++) {
      if (values1[i] != values2[i]) {
	return false;
      }
    }
    return true;
  } 
  

  /**
   * internal method that returns the distance between an instance and a cluster
   */
  protected double distance (Instance instance, Cluster cluster) throws Exception {
    Integer idx;
    double distance = 0;
    
    switch (m_linkingType) {
    case SINGLE_LINK:
      double minDistance = Double.MAX_VALUE;
      for (int i = 0; i < cluster.size(); i++) {
	Instance clusterInstance = (Instance) cluster.get(i);
	double currDistance = m_metric.distance(instance, clusterInstance);
	if (currDistance < minDistance)
	  minDistance = currDistance;
      }
      distance = minDistance;
      break;
    case COMPLETE_LINK:
      double maxDistance = Double.MIN_VALUE;
      for (int i = 0; i < cluster.size(); i++) {
	Instance clusterInstance = (Instance) cluster.get(i);
	double currDistance = m_metric.distance(instance, clusterInstance);
	if (currDistance > maxDistance)
	  maxDistance = currDistance;
      }
      distance = maxDistance;
      break;
    case GROUP_AVERAGE:
      double avgDistance = 0;
      for (int i = 0; i < cluster.size(); i++) {
	Instance clusterInstance = (Instance) cluster.get(i);
	avgDistance += m_metric.distance(instance, clusterInstance);
      }
      distance = avgDistance/cluster.size();
      break;
    default:
      throw new Exception("Unknown linkage type!");
    }
    return distance;
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
   * @return verbose messages on(true) or off (false)
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
    
    Vector newVector = new Vector(2);
    
    newVector.addElement(new Option("\tThreshold.\n"
				    +"\t(default=MAX_DOUBLE)", "T", 1,"-T <0-MAX_DOUBLE>"));
    newVector.addElement(new Option("\tNumber of clusters.\n"
				    +"a\t(default=-1)", "N", 1,"-N <-1-MAX_INT100%>"));
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   *
   * Valid options are:<p>
   *
   * -A <0-100> <br>
   * Acuity. <p>
   *
   * -C <0-100> <br>
   * Cutoff. <p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   *
   **/
  public void setOptions(String[] options) throws Exception {
    String optionString;

    optionString = Utils.getOption('N', options); 
    if (optionString.length() != 0) {
      setNumClusters(Integer.parseInt(optionString));
    }
    
  }


  /**
   * Gets the current settings of Greedy Agglomerative Clustering
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {
    
    String [] options = new String [70];
    int current = 0;
    options[current++] = "-N"; 
    options[current++] = "" + m_numClusters;

    if (m_linkingType == SINGLE_LINK) {
      options[current++] = "-I";
    } else if (m_linkingType == COMPLETE_LINK) {
      options[current++] = "-C";
    } else if (m_linkingType == GROUP_AVERAGE) {
      options[current++] = "-G";
    }

    if (m_seedable) {
      options[current++] = "-S";
    }

    options[current++] = "-M";
    options[current++] = m_metric.getClass().getName();
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
 
  /**
   * Train the clusterer using specified parameters
   *
   * @param instances Instances to be used for training
   */
  public void trainClusterer (Instances instances) throws Exception {
    if (m_metric instanceof LearnableMetric) {
      if (((LearnableMetric)m_metric).getTrainable()) {
	((LearnableMetric)m_metric).learnMetric(instances);
      }
      else {
	throw new Exception ("Metric is not trainable");
      }
    }
    else {
      throw new Exception ("Metric is not trainable");
    }
  }

  /** returns objective function, needed for compatibility with SemiSupClusterer */
  public double objectiveFunction() {
    return Double.NaN;
  }

    /** Return the number of clusters */
  public int getNumClusters() {
    return m_numClusters;
  }

  /** A duplicate function to conform to Clusterer abstract class.
   * @returns the number of clusters
   */
  public int numberOfClusters() {
    return getNumClusters();
  }

  /**
   * get an array of random indeces out of n possible values.
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


  
  // Main method for testing this class
  public static void main(String [] argv)  {
    try {
      //////// Iris data
      //String datafile = "/u/ml/software/weka-latest/data/iris.arff";
      //      String datafile = "/u/mbilenko/ml/tivoli/user-features-GroupClassGrad.arff";
      String datafile = "/u/mbilenko/ml/tivoli/data/user-features-processClass.arff";
      //      String datafile = "/u/mbilenko/weka/data/glass.arff";
    
      // set up the data
      FileReader reader = new FileReader (datafile);
      Instances data = new Instances (reader);



      // filter out bad attributes for tivoli clustering
      String [] filteredProcesses = {"pico", "twm", "Xvnc", "lpr", "fvwm2", "xclock", "FvwmButtons", "FvwmPager", "ymessenger.bin",
      "vim", "vi", "xemacs", "xscreensaver", "gnome-panel", "gnome-settings-daemon", "gconfd-2", "xlock", "kdesud", "ssh",
      "tasklist_applet", "panel", "gnome-session", "gnome-smproxy", "MozillaFirebird-bin", "nautilus", "mutt",
      "mixer_applet2", "metacity", "bonobo-activation-server", "csh", "nautilus-throbber", "xmms", "realplay", "konqueror", "knode", "kdesktop_lock", "kwrapper", "artsd", "esd", "gnome-panel", "gnome-terminal", "mail", "gnome-name-service", "deskguide_applet", "sawfish",
      "gaim", "konsole", "opera", "enlightenment", "6", "wmaker"};
      System.out.println("filtered=" + filteredProcesses.length); 
      int[] descrIndeces = new int[filteredProcesses.length];
      for (int i = 0; i < descrIndeces.length; i++) {
	Attribute attr = data.attribute(filteredProcesses[i]);
	System.out.println(i + ": " + attr);
	descrIndeces[i] = attr.index();
      }

      Remove attributeFilter = new Remove();
      attributeFilter.setAttributeIndicesArray(descrIndeces);
      attributeFilter.setInvertSelection(false);
      attributeFilter.setInputFormat(data);
      data = Filter.useFilter(data, attributeFilter);
      
      // Make the last attribute be the class 
      int theClass = data.numAttributes();
      data.setClassIndex(theClass-1); // starts with 0
      //        int numClusters = data.numClasses();
      
      Instances clusterData = new Instances(data);
      clusterData.deleteClassAttribute();
      

      WeightedEuclidean euclidean = new WeightedEuclidean(clusterData.numAttributes());
      WeightedDotP dotp = new WeightedDotP(clusterData.numAttributes());
      //      HAC hac = new HAC(euclidean);
      HAC hac = new HAC(dotp);
      hac.setVerbose(false);
      clusterData = hac.filterInstanceDescriptions(clusterData);

      // cluster without seeding
      System.out.println("\nClustering the user data ...\n");      
      hac.setLinkingType(new SelectedTag(COMPLETE_LINK, TAGS_LINKING));

      // trim the instances
      //      int i = 6;
      // while  (i < clusterData.numInstances()) {
      //	clusterData.delete(i);
      //}

      // cluster with seeding      
      //      ArrayList seedArray = new ArrayList();
      //      for (int i = 0; i < 19; i++) {
      //	seedArray.add(clusterData.instance(i));
      //      }
//        seedArray.add(clusterData.instance(0));
//        seedArray.add(clusterData.instance(1));
//        seedArray.add(clusterData.instance(2));
//        seedArray.add(clusterData.instance(3));
//        seedArray.add(clusterData.instance(4));      

//        seedArray.add(clusterData.instance(50));
//        seedArray.add(clusterData.instance(51));
//        seedArray.add(clusterData.instance(52));
//        seedArray.add(clusterData.instance(53));
//        seedArray.add(clusterData.instance(54));

//        seedArray.add(clusterData.instance(100));
//        seedArray.add(clusterData.instance(101));
//        seedArray.add(clusterData.instance(102));
//        seedArray.add(clusterData.instance(103));
//        seedArray.add(clusterData.instance(104));

//        Seeder seeder = new Seeder(clusterData, data);
//        seeder.setVerbose(false);
//        seeder.createSeeds(seedArray);
//        HashMap seedHash = seeder.getSeeds();

//        hac.setSeedHash(seedHash);


      HashMap classInstanceHash = new HashMap();
      // get the data for each class
      for (int i = 0; i < data.numInstances(); i++) {
	Instance instance = data.instance(i);
	Integer classValue = new Integer((int) instance.classValue());
	if (classInstanceHash.containsKey(classValue)) {
	  ArrayList classList = (ArrayList) classInstanceHash.get(classValue);
	  classList.add(new Integer(i));
	  System.out.println("Seen class; now has " + classList.size() + " elements");
	} else { // unseen class
	  System.out.println("Unseen class " + classValue);
	  ArrayList classList = new ArrayList();
	  classList.add(new Integer(i));
	  classInstanceHash.put(classValue, classList);
	} 
      }

      // sample from the classes that have more than 1 instance
      double seedProportion = 0.7;
      ArrayList seedArray = new ArrayList();
      Iterator iterator = classInstanceHash.entrySet().iterator();
      while (iterator.hasNext()) {
	Map.Entry entry = (Map.Entry) iterator.next();
	ArrayList classList = (ArrayList) entry.getValue();	
	System.out.println("Classlist for " + entry.getKey() + " has " + classList.size() + " elements\n");		
	if (classList.size() > 1) {
	  int [] seedIndeces = randomSubset((int) ((classList.size() + 0.0) * seedProportion), classList.size());
	  System.out.println("Seeding for class " + entry.getKey() + " using " + seedIndeces.length);
	  for (int i = 0; i < seedIndeces.length; i++) {
	    seedArray.add(clusterData.instance(((Integer)(classList.get(seedIndeces[i]))).intValue()));
	    System.out.println("Adding seed " + classList.get(seedIndeces[i]));
	  } 
	}
      }
      Seeder seeder = new Seeder(clusterData, data);
      seeder.setVerbose(false);
      seeder.createSeeds(seedArray);
      HashMap seedHash = seeder.getSeeds();
      hac.setSeedHash(seedHash);
      
      hac.buildClusterer(clusterData, 1);
      hac.printClusters();

//        System.out.println("Cluster assignments: ");
//        for (int i=0; i < hac.m_clusterAssignments.length; i++) {
//  	System.out.print(i + ":" + hac.m_clusterAssignments[i] + "  ");
//        }

//        System.out.println("\n\n");
//        for (int j = 0; j < clusterData.numInstances(); j++) {
//  	System.out.println(j + ":" + hac.clusterInstance(clusterData.instance(j)));
//        }


      ////////////////////////////////////////////////////
      //  HI-DIM TESTING
      ////////////////////////////////////////////////////
      //////// Text data - 300 documents

//        datafile = "/u/ml/software/weka-latest/data/20newsgroups/different-100_fromCCS.arff";
//        System.out.println("\nClustering diff-100 newsgroup data with seeding, using constrained HAC...\n");      

//        // set up the data
//        reader = new FileReader (datafile);
//        data = new Instances (reader);
//        System.out.println("Initial data has size: " + data.numInstances());

//        // Make the last attribute be the class 
//        theClass = data.numAttributes();
//        data.setClassIndex(theClass-1); // starts with 0
//        numClusters = data.numClasses();
      
//        WeightedDotP dotp = new WeightedDotP(data.numAttributes());
//        hac = new HAC (dotp);

//        // cluster with seeding      
//        Instances seeds = new Instances(data, 0, 5);
//        seeds.add(data.instance(100));
//        seeds.add(data.instance(101));
//        seeds.add(data.instance(102));
//        seeds.add(data.instance(103));
//        seeds.add(data.instance(104));
//        seeds.add(data.instance(200));
//        seeds.add(data.instance(201));
//        seeds.add(data.instance(202));
//        seeds.add(data.instance(203));
//        seeds.add(data.instance(204));

//        System.out.println("Labeled data has size: " + seeds.numInstances() + ", number of attributes: " + data.numAttributes());

//        data.delete(204);
//        data.delete(203);
//        data.delete(202);
//        data.delete(201);
//        data.delete(200);
//        data.delete(104);
//        data.delete(103);
//        data.delete(102);
//        data.delete(101);
//        data.delete(100);
//        data.delete(4);
//        data.delete(3);
//        data.delete(2);
//        data.delete(1);
//        data.delete(0);

//        System.out.println("Unlabeled data has size: " + data.numInstances());

//        // Remove the class labels before clustering

//        clusterData = new Instances(data);
//        //      clusterData.deleteAttributeAt(theClass-1);
//        clusterData.deleteClassAttribute();

//        hac.setVerbose(false);
//        hac.setSeedable(true);
//        hac.buildClusterer(seeds, clusterData, theClass, numClusters);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
