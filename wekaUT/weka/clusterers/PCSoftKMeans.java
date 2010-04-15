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
 *    PCSoftKMeans.java
 *    Copyright (C) 2003 Sugato Basu 
 *
 */
package weka.clusterers;

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.core.metrics.*;
import  weka.filters.Filter;
import  weka.filters.unsupervised.attribute.Remove;

/**
 * Pairwise constrained k means clustering class.
 *
 * Valid options are:<p>
 *
 * -N <number of clusters> <br>
 * Specify the number of clusters to generate. <p>
 *
 * -R <random seed> <br>
 * Specify random number seed <p>
 *
 * -A <algorithm> <br>
 * The algorithm can be "Simple" (simple KMeans) or "Spherical" (spherical KMeans)
 *
 * -M <metric-class> <br>
 * Specifies the name of the distance metric class that should be used
 * 
 * .... etc. 
 *
 * @author Sugato Basu(sugato@cs.utexas.edu)
 * @see Clusterer
 * @see OptionHandler
 */
public class PCSoftKMeans extends DistributionClusterer implements OptionHandler,SemiSupClusterer,ActiveLearningClusterer {

  /** Name of clusterer */
  String m_name = "PCSoftKMeans";

  /** holds the instances in the clusters */
  protected ArrayList m_Clusters = null;

  /** holds the instance indices in the clusters, mapped to their
      probabilities 
  */
  protected HashMap[] m_IndexClusters = null;
  
  /** holds the ([instance pair] -> [type of constraint])
      mapping. Note that the instance pairs stored in the hash always
      have constraint type InstancePair.DONT_CARE_LINK, the actual
      link type is stored in the hashed value 
  */
  
  protected HashMap m_ConstraintsHash = null;

  /** adjacency list for neighborhoods */
  protected HashSet[] m_AdjacencyList;

  /** colors required for keeping track of DFS visit */
  final int WHITE = 300;
  final int GRAY = 301;
  final int BLACK = 302;

  /** holds the points involved in the constraints */
  protected HashSet m_SeedHash = null;

  /** weight to be given to each constraint */
  protected double m_CannotLinkWeight = 1;

  /** weight to be given to each constraint */
  protected double m_MustLinkWeight = 1;

  /** kappa value for vmf distribution */
  protected double m_Kappa = 2;

  /** max kappa value for vmf distribution */
  protected double m_MaxKappaSim = 100;
  protected double m_MaxKappaDist = 10;

  /** the maximum number of cannot-link constraints allowed */
  protected final static int m_MaxConstraintsAllowed = 10000;

  /** verbose? */
  protected boolean m_verbose = false;

  /** distance Metric */
  protected Metric m_metric = new WeightedEuclidean();

  /** has the metric has been constructed?  a fix for multiple buildClusterer's */
  protected boolean m_metricBuilt = false;

  /** indicates whether instances are sparse */
  protected  boolean m_isSparseInstance = false;

  /** Is the objective function increasing or decreasing?  Depends on type
      of metric used:  for similarity-based metric - increasing, 
      for distance-based - decreasing 
  */
  protected boolean m_objFunDecreasing = true;

  /** Seedable or not (true by default) */
  protected boolean m_Seedable = true;

  /** keep track of the number of iterations completed before convergence
   */
  protected int m_Iterations = 0;
  
  /** Define possible algorithms */
  public static final int ALGORITHM_SIMPLE = 1;
  public static final int ALGORITHM_SPHERICAL = 2;
  public static final Tag[] TAGS_ALGORITHM = {
    new Tag(ALGORITHM_SIMPLE, "Simple"),
    new Tag(ALGORITHM_SPHERICAL, "Spherical")
      };
  
  /** algorithm, by default spherical */
  protected int m_Algorithm = ALGORITHM_SIMPLE;
  
  /** min. absolute difference of objective function values for
      convergence 
  */
  protected double m_ObjFunConvergenceDifference = 1e-3; // difference less significant than 3rd place of decimal

  /** value of objective function */
  protected double m_Objective;

  /** returns objective function */
  public double objectiveFunction() {
    return m_Objective;
  }

  /**
   * training instances with labels
   */
  protected Instances m_TotalTrainWithLabels;

  /**
   * training instances
   */
  protected Instances m_Instances;


  /** A hash where the instance checksums are hashed */
  protected HashMap m_checksumHash = null; 
  protected double []m_checksumCoeffs = null; 


  /** test data -- required to make sure that test points are not
      selected during active learning */
  protected int m_StartingIndexOfTest = -1;

  /**
   * number of clusters to generate, default is -1 to get it from labeled data
   */
  protected int m_NumClusters = 3;


  /** Number of clusters in the process*/
  protected int m_NumCurrentClusters = 0; 

  /**
   * m_FastMode = true  => fast computation of meanOrMode in centroid calculation, useful for high-D data sets
   * m_FastMode = false => usual computation of meanOrMode in centroid calculation
   */
  protected boolean m_FastMode = true;

  /**
   * holds the cluster centroids
   */
  protected Instances m_ClusterCentroids;


  /**
   * holds the global centroids
   */
  protected Instance m_GlobalCentroid;

  /**
   * holds the default perturbation value for randomPerturbInit
   */
  protected double m_DefaultPerturb = 0.7;

  /**
   * holds the default merge threshold for matchMergeStep
   */
  protected double m_MergeThreshold = 0.15;

  /**
   * temporary variable holding posterior cluster distribution of
   * points while iterating 
   */
  protected double [][] m_ClusterDistribution;

  /**
   * temporary variable holding cluster assignments while iterating
   */
  protected int [] m_ClusterAssignments;

  /**
   * temporary variable holding cluster sums while iterating
   */
  protected Instance [] m_SumOfClusterInstances;

  /**
   * holds the random Seed, useful for randomPerturbInit
   */
  protected int m_RandomSeed = 42;

  /** neighbor list: points in each neighborhood inferred from
      constraints */
  protected HashSet[] m_NeighborSets;

  /** assigned set for active learning: whether a point has been
      assigned or not */
  HashSet m_AssignedSet;

  /* Constructor */
  public PCSoftKMeans() {
  }

  /* Constructor */
  public PCSoftKMeans(Metric metric) {
    m_metric = metric;
    m_objFunDecreasing = metric.isDistanceBased();
  }

  /**
   * We always want to implement SemiSupClusterer from a class
   * extending Clusterer.  We want to be able to return the underlying
   * parent class.
   * @return parent Clusterer class */
  public Clusterer getThisClusterer() {
    return this;
  }
 
  /**
   * Generates a clusterer. Instances in data have to be either all
   * sparse or all non-sparse
   *
   * @param data set of instances serving as training data
   * @exception Exception if the clusterer has not been 
   * generated successfully */
  public void buildClusterer(Instances data) throws Exception {
    System.out.println("Must link weight: " + m_MustLinkWeight);
    System.out.println("Cannot link weight: " + m_CannotLinkWeight);

    setInstances(data);

    // Don't rebuild the metric if it was already trained
    if (!m_metricBuilt) {
      m_metric.buildMetric(data.numAttributes());
    }
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    m_ClusterAssignments = new int [m_Instances.numInstances()];

    if (m_Instances.checkForNominalAttributes() && m_Instances.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle nominal attributes\n");
    }

    System.out.println("Initializing clustering ...");
    
    nonActivePairwiseInit();
    System.out.println("Done initializing clustering ...");

    if (m_Seedable) {
      //      System.out.println("Initial assignments of seed points:");
      //      printIndexClusters();
    }
    if (m_verbose) {
      for (int i=0; i<m_NumClusters; i++) {
	System.out.println("Centroid " + i + ": " + m_ClusterCentroids.instance(i));
      }
    }
    runEM();
  }

  /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using labeledData as seeds
   *
   * @param labeledData labeled instances to be used as seeds
   * @param unlabeledData unlabeled instances
   * @param classIndex attribute index in labeledData which holds class info
   * @param numClusters number of clusters
   * @param startingIndexOfTest from where test data starts in
   *                            unlabeledData, useful if clustering is transductive
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(Instances labeledData, Instances unlabeledData, int classIndex, int numClusters, int startingIndexOfTest) throws Exception {

    // !!!! Dummy function, for compatibility with interface !!!!

    throw new Exception("Not implemented for PCSoftKMeans");
  }

  /**
   * Cluster given instances to form the specified number of clusters.
   *
   * @param data instances to be clustered
   * @param num_clusters number of clusters to create
   * @exception Exception if something goes wrong.
   */
  public void buildClusterer(Instances data, int num_clusters) throws Exception {
    m_NumClusters = num_clusters;
    if (m_Algorithm == ALGORITHM_SPHERICAL && m_metric instanceof WeightedDotP) {
      ((WeightedDotP)m_metric).setLengthNormalized(false); // since instances and clusters are already normalized, we don't need to normalize again while computing similarity - saves time
    }
    if (data.instance(0) instanceof SparseInstance) {
      m_isSparseInstance = true;
    }    
    buildClusterer(data);
  }

  /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using labeledData as seeds
   *
   * @param labeledTrainPairs labeled instances to be used as seeds
   * @param unlabeledData unlabeled training (+ test for transductive) instances
   * @param labeledTrain labeled training instances
   * @param startingIndexOfTest starting index of test set in unlabeled data
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(ArrayList labeledPair, Instances unlabeledData, Instances labeledTrain, int startingIndexOfTest) throws Exception {
    int classIndex = labeledTrain.numAttributes(); // assuming that the last attribute is always the class
    m_TotalTrainWithLabels = labeledTrain;

    if (labeledPair != null) {
      m_SeedHash = new HashSet((int) (unlabeledData.numInstances()/0.75 + 10)) ;
      m_ConstraintsHash = new HashMap(m_MaxConstraintsAllowed); 

      for (int i=0; i<labeledPair.size(); i++) {
	InstancePair pair = (InstancePair) labeledPair.get(i);
	Integer firstInt = new Integer(pair.first);
	Integer secondInt = new Integer(pair.second);
	// for first point 
	if(!m_SeedHash.contains(firstInt)) { // add instances with constraints to seedHash
	  m_SeedHash.add(firstInt);
	}
	
	// for second point 
	if(!m_SeedHash.contains(secondInt)) {
	  m_SeedHash.add(secondInt);
	}
	if (pair.first >= pair.second) {
	  throw new Exception("Ordering reversed - something wrong!!");
	} 
	else {
	  InstancePair newPair = new InstancePair(pair.first, pair.second, InstancePair.DONT_CARE_LINK);
	  m_ConstraintsHash.put(newPair, new Integer(pair.linkType)); // WLOG first < second
	}
      }
    }

    // normalize all data for SPKMeans
    if (m_Algorithm == ALGORITHM_SPHERICAL) {
      for (int i=0; i<unlabeledData.numInstances(); i++) {
	normalize(unlabeledData.instance(i));
      }
    }
    m_StartingIndexOfTest = startingIndexOfTest;
    if (m_verbose) {
      System.out.println("Starting index of test: " + m_StartingIndexOfTest);
    }

    // learn metric using labeled data, then cluster both the labeled and unlabeled data
    m_metric.buildMetric(unlabeledData.numAttributes());
    m_metricBuilt = true;
    buildClusterer(unlabeledData, labeledTrain.numClasses());
  }

  /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using labeledData as seeds -- NOT USED FOR PCSoftKMeans!!!
   *
   * @param labeledData labeled instances to be used as seeds
   * @param unlabeledData unlabeled instances
   * @param classIndex attribute index in labeledData which holds class info
   * @param numClusters number of clusters
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(Instances labeledData, Instances unlabeledData, int classIndex, int numClusters) throws Exception {

    /// ----- NOT USED FOR PCSoftKMeans!!! ----- ///

    if (m_Algorithm == ALGORITHM_SPHERICAL) {
      for (int i=0; i<labeledData.numInstances(); i++) {
	normalize(labeledData.instance(i));
      }
      for (int i=0; i<unlabeledData.numInstances(); i++) {
	normalize(unlabeledData.instance(i));
      }
    }
    // remove labels of labeledData before putting in seedHash
    Instances clusterData = new Instances(labeledData);
    clusterData.deleteClassAttribute();

    // create seedHash from labeledData
    if (m_Seedable) {
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
    buildClusterer(clusterData, numClusters);
  }

  /**
   * Reset all values that have been learned
   */
  public void resetClusterer()  throws Exception{
    if (m_metric instanceof LearnableMetric) {
      ((LearnableMetric)m_metric).resetMetric();
    }
    m_SeedHash = null;
    m_ConstraintsHash = null;
  }


  /** Set default perturbation value
   * @param p perturbation fraction
   */
  public void setDefaultPerturb(double p) {
    m_DefaultPerturb = p;
  }


  /** Get default perturbation value
   * @return perturbation fraction
   */
  public double getDefaultPerturb(){
    return m_DefaultPerturb;
  }


  /** Turn seeding on and off
   * @param seedable should seeding be done?
   */
  public void setSeedable(boolean seedable) {
    m_Seedable = seedable;
  }

  /** Is seeding performed?
   * @return is seeding being done?
   */
  public boolean getSeedable() {
      return m_Seedable;
  }
  
  /**
   * We can have clusterers that don't utilize seeding
   */
  public boolean seedable() {
    return m_Seedable;
  }

  /** Creates the global cluster centroid */
  protected void createCentroids() throws Exception {
    // initialize using m_NumCurrentClusters neighborhoods (< m_NumClusters), make random for rest

    System.out.println("Creating centroids");
    if (m_verbose)
      System.out.println("Current number of clusters: " + m_NumCurrentClusters);

    // compute centroids of all clusters
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    for (int i=0; i<m_NumCurrentClusters; i++) {
      if (m_SumOfClusterInstances[i] != null) {
	if (m_verbose) {
	  System.out.println("Normalizing cluster center " + i);
	}
	if (!m_objFunDecreasing) {
	  normalize(m_SumOfClusterInstances[i]);
	}
	else {
	  normalizeByWeight(m_SumOfClusterInstances[i]);
	}
      }
      m_SumOfClusterInstances[i].setDataset(m_Instances);
      m_ClusterCentroids.add(m_SumOfClusterInstances[i]);
    }

    // fill up remaining by randomPerturbInit
    if (m_NumCurrentClusters < m_NumClusters) {
      
      // find global centroid
      System.out.println("Creating global centroid");
      double [] globalValues = new double[m_Instances.numAttributes()];
      if (m_FastMode && m_isSparseInstance) {
	globalValues = meanOrMode(m_Instances); // uses fast meanOrMode
      }
      else {
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  globalValues[j] = m_Instances.meanOrMode(j); // uses usual meanOrMode
	}
      }
      
      // global centroid is dense in SPKMeans
      m_GlobalCentroid = new Instance(1.0, globalValues);
      m_GlobalCentroid.setDataset(m_Instances);
      Random random = new Random(m_RandomSeed);

      // normalize before random perturbation
      if (!m_objFunDecreasing) {
	normalizeInstance(m_GlobalCentroid);
      }
      
      System.out.println("Creating " + (m_NumClusters - m_NumCurrentClusters) + " random centroids");
      for (int i=m_NumCurrentClusters; i<m_NumClusters; i++) {
	double [] values = new double[m_Instances.numAttributes()];
	double normalizer = 0;
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  values[j] = m_GlobalCentroid.value(j) * (1 + m_DefaultPerturb * (random.nextFloat() - 0.5));
	  normalizer += values[j] * values[j];
	}
	if (!m_objFunDecreasing) {
	  normalizer = Math.sqrt(normalizer);
	  for (int j = 0; j < m_Instances.numAttributes(); j++) {
	    values[j] /= normalizer;
	  }
	}

	// values suitably normalized at this point if required
	if (m_isSparseInstance) {
	  m_ClusterCentroids.add(new SparseInstance(1.0, values)); // sparse for consistency with other cluster centroids
	}
	else {
	  m_ClusterCentroids.add(new Instance(1.0, values));
	}
      }
    }
    System.out.println("Finished creating centroids");
    m_NumCurrentClusters = m_NumClusters;
  }

  /** adding other inferred ML and CL links to m_ConstraintsHash, from
   *   m_NeighborSets 
   */
  protected void addMLAndCLTransitiveClosure(int[] indices) throws Exception {
    // add all ML links within neighborhoods selected as clusters
    if (m_verbose) {
      for (int j=0; j<m_NumCurrentClusters; j++) {
	int i = j;
	if (indices != null) {
	  i = indices[j];
	}
	System.out.println("Neighborhood list " + j + " is:");
	System.out.println(m_NeighborSets[i]);
      }
    }

    for (int j=0; j<m_NumCurrentClusters; j++) {
      int i = j;
      if (indices != null) {
	i = indices[j];
      }
      if (m_NeighborSets[i] != null) {
	Iterator iter1 = m_NeighborSets[i].iterator();
	while (iter1.hasNext()) {
	  int first = ((Integer) iter1.next()).intValue();
	  Iterator iter2 = m_NeighborSets[i].iterator();
	  while (iter2.hasNext()) {
	    int second = ((Integer) iter2.next()).intValue();
	    if (first < second) {
	      InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
	      if (!m_ConstraintsHash.containsKey(pair)) {
		m_ConstraintsHash.put(pair, new Integer(InstancePair.MUST_LINK));
		if (m_verbose) {
		  System.out.println("Adding inferred ML link: " + pair);
		}
		if (!m_SeedHash.contains(new Integer(first))) {
		  m_SeedHash.add(new Integer(first));
		}
		if (!m_SeedHash.contains(new Integer(second))) {
		  m_SeedHash.add(new Integer(second));
		}
	      }
	    }
	  }
	}
      }
    }

    // add all CL links between clusters
    for (int ii=0; ii<m_NumCurrentClusters; ii++) {
      int i = ii;
      if (indices != null) {
	i = indices[ii];
      }
      if (m_NeighborSets[i] != null) {
	Iterator iter1 = m_NeighborSets[i].iterator();
	while (iter1.hasNext()) {
	  int index1 = ((Integer) iter1.next()).intValue();
	  for (int jj=ii+1; jj<m_NumCurrentClusters; jj++) {
	    int j = jj;
	    if (indices != null) {
	      j = indices[jj];
	    }
	    if (m_NeighborSets[j] != null) {
	      Iterator iter2 = m_NeighborSets[j].iterator();
	      while (iter2.hasNext()) {
		int index2 = ((Integer) iter2.next()).intValue();
		int first = (index1 < index2)? index1:index2;
		int second = (index1 >= index2)? index1:index2;
		if (first == second) {
		  throw new Exception(" Same instance " + first + " cannot be in cluster: " + i + " and cluster " + j);
		}
		InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
		if (!m_ConstraintsHash.containsKey(pair)) {
		  m_ConstraintsHash.put(pair, new Integer(InstancePair.CANNOT_LINK));
		  if (m_verbose) {
		    System.out.println("Adding inferred CL link: " + pair);
		  }
		  if (!m_SeedHash.contains(new Integer(first))) {
		    m_SeedHash.add(new Integer(first));
		  }
		  if (!m_SeedHash.contains(new Integer(second))) {
		    m_SeedHash.add(new Integer(second));
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  /** Main Depth First Search routine */
  protected void DFS() throws Exception {
    int [] vertexColor = new int[m_Instances.numInstances()];
    m_NumCurrentClusters = 0;

    for(int u=0; u<m_Instances.numInstances(); u++){
      vertexColor[u] = WHITE;
    }
    
    for(int u=0; u<m_Instances.numInstances(); u++){
      if (m_AdjacencyList[u] != null && vertexColor[u] == WHITE) {
  	m_NeighborSets[m_NumCurrentClusters] = new HashSet();
	DFS_VISIT(u, vertexColor); 	// finds whole neighbourhood of u
	m_NumCurrentClusters++;
      }
    }
  }

  /** Recursive subroutine for DFS */
  protected void DFS_VISIT(int u, int[] vertexColor) throws Exception {
    vertexColor[u] = GRAY;
    
    Iterator iter = null;
    if (m_AdjacencyList[u] != null) {
      iter = m_AdjacencyList[u].iterator();
      while (iter.hasNext()) {
	int j = ((Integer) iter.next()).intValue();
	if(vertexColor[j] == WHITE){ // if the vertex is still undiscovered
	  DFS_VISIT(j, vertexColor);
	}
      }
    }
    // update stats for u
    m_ClusterAssignments[u] = m_NumCurrentClusters;
    m_NeighborSets[m_NumCurrentClusters].add(new Integer(u));
    m_SumOfClusterInstances[m_NumCurrentClusters] = sumWithInstance(m_SumOfClusterInstances[m_NumCurrentClusters], m_Instances.instance(u));
    vertexColor[u] = BLACK;
  }


  /** Initialization routine for non-active algorithm */
  protected void nonActivePairwiseInit() throws Exception  {
    m_NeighborSets = new HashSet[m_Instances.numInstances()];
    m_SumOfClusterInstances = new Instance[m_Instances.numInstances()];
    m_AdjacencyList = new HashSet[m_Instances.numInstances()];
    
    for (int i=0; i<m_Instances.numInstances(); i++) {
      m_ClusterAssignments[i] = -1;
    }

    if (m_ConstraintsHash != null) {
      Set pointPairs = (Set) m_ConstraintsHash.keySet(); 
      Iterator pairItr = pointPairs.iterator();
      System.out.println("In non-active init");
      
      // iterate over the pairs in ConstraintHash to create Adjacency List
      while( pairItr.hasNext() ){
	InstancePair pair = (InstancePair) pairItr.next();
	int linkType = ((Integer) m_ConstraintsHash.get(pair)).intValue();
	if (m_verbose)
	  System.out.println(pair + ": type = " + linkType);
	if( linkType == InstancePair.MUST_LINK ){ // concerned with MUST-LINK in Adjacency List
	  if (m_AdjacencyList[pair.first] == null) {
	    m_AdjacencyList[pair.first] = new HashSet();
	  }
	  if (!m_AdjacencyList[pair.first].contains(new Integer(pair.second))) {
	    m_AdjacencyList[pair.first].add(new Integer(pair.second));
	  }
	  
	  if (m_AdjacencyList[pair.second] == null) {
	    m_AdjacencyList[pair.second] = new HashSet();
	  }
	  if (!m_AdjacencyList[pair.second].contains(new Integer(pair.first))) {
	    m_AdjacencyList[pair.second].add(new Integer(pair.first));
	  }
	}
      }
      
      // DFS for finding connected components in Adjacency List, updates required stats
      DFS();
    }
    
    if (!m_Seedable) { // don't perform any seeding, initialize from random
      m_NumCurrentClusters = 0;
    }

    // if the required number of clusters has been obtained, wrap-up
    if( m_NumCurrentClusters >= m_NumClusters ){
      if (m_verbose) {
	System.out.println("Got the required number of clusters ...");
	System.out.println("num clusters: " + m_NumClusters + ", num current clusters: " + m_NumCurrentClusters);
      }
      int clusterSizes[] = new int[m_NumCurrentClusters];
      for (int i=0; i<m_NumCurrentClusters; i++) {
	if (m_verbose) {
	  System.out.println("Neighbor set: " + i + " has size: " + m_NeighborSets[i].size());
	}
	clusterSizes[i] = -m_NeighborSets[i].size(); // for reverse sort (decreasing order)
      }	
      int[] indices = Utils.sort(clusterSizes);
      
      Instance[] clusterCentroids = new Instance[m_NumClusters];

      // compute centroids of m_NumClusters clusters
      m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
      for (int j=0; j < m_NumClusters; j++) {
	int i = indices[j];
	if (m_SumOfClusterInstances[i] != null) {
	  if (m_verbose) {
	    System.out.println("Normalizing instance " + i);
	  }
	  if (!m_objFunDecreasing) {
	    normalize(m_SumOfClusterInstances[i]);
	  }
	  else {
	    normalizeByWeight(m_SumOfClusterInstances[i]);
	  }
	}
	Iterator iter = m_NeighborSets[i].iterator();
	while (iter.hasNext()) { // assign points of new cluster
	  int instNumber = ((Integer) iter.next()).intValue();
	  if (m_verbose) {
	    System.out.println("Assigning " + instNumber + " to cluster: " + j);
	  }
	  // have to re-assign after sorting
	  m_ClusterAssignments[instNumber] = j;
	}

	m_SumOfClusterInstances[j].setDataset(m_Instances);
	// m_SumOfClusterInstances suitably normalized now
	m_ClusterCentroids.add(m_SumOfClusterInstances[i]);
      }
      for (int j=m_NumClusters; j < m_NumCurrentClusters; j++) {
	int i = indices[j];
	Iterator iter = m_NeighborSets[i].iterator();
	while (iter.hasNext()) {
	  int instNumber = ((Integer) iter.next()).intValue();
	  if (m_verbose) {
	    System.out.println("Assigning " + instNumber + " to cluster -1");
	  }
	  m_ClusterAssignments[instNumber] = -1;
	}
      }
      m_NumCurrentClusters = m_NumClusters;
      // adding other inferred ML and CL links
      addMLAndCLTransitiveClosure(indices);
      return;
    }
    else if( m_NumCurrentClusters < m_NumClusters ){
      createCentroids();
      addMLAndCLTransitiveClosure(null);
    }
  }


  /** This function divides every attribute value in an instance by
   *  the instance weight -- useful to find the mean of a cluster in
   *  Euclidean space 
   *  @param inst Instance passed in for normalization (destructive update)
   */
  protected void normalizeByWeight(Instance inst) {
    double weight = inst.weight();
    if (m_verbose) {
      System.out.println("Before weight normalization: " + inst);
    }
    if (inst instanceof SparseInstance) { 
      for (int i=0; i<inst.numValues(); i++) {
	inst.setValueSparse(i, inst.valueSparse(i)/weight);
      }
    }
    else if (!(inst instanceof SparseInstance)) {
      for (int i=0; i<inst.numAttributes(); i++) {
	inst.setValue(i, inst.value(i)/weight);
      }
    }
    if (m_verbose) {
      System.out.println("After weight normalization: " + inst);
    }
  }


  /** Finds the sum of instance sum with instance inst 
   */
  Instance sumWithInstance(Instance sum, Instance inst) throws Exception {
    Instance newSum;
    if (sum == null) {
      if (m_isSparseInstance) {
	newSum = new SparseInstance(inst);
	newSum.setDataset(m_Instances);
      }
      else {
	newSum = new Instance(inst);
	newSum.setDataset(m_Instances);
      }
    }
    else {
      newSum = sumInstances(sum, inst);
    }
    return newSum;
  }


  /** Finds sum of 2 instances (handles sparse and non-sparse)
   */

  protected Instance sumInstances(Instance inst1, Instance inst2) throws Exception {
    int numAttributes = inst1.numAttributes();
    if (inst2.numAttributes() != numAttributes) {
      throw new Exception ("Error!! inst1 and inst2 should have same number of attributes.");
    }
    //      if (m_verbose) {
    //        System.out.println("Instance 1 is: " + inst1 + ", instance 2 is: " + inst2);
    //      }
    double weight1 = inst1.weight(), weight2 = inst2.weight();
    double [] values = new double[numAttributes];
    Instance newInst;
    
    for (int i=0; i<numAttributes; i++) {
      values[i] = 0;
    }
    
    if (inst1 instanceof SparseInstance && inst2 instanceof SparseInstance) {
      for (int i=0; i<inst1.numValues(); i++) {
	int indexOfIndex = inst1.index(i);
	values[indexOfIndex] = inst1.valueSparse(i);
      }
      for (int i=0; i<inst2.numValues(); i++) {
	int indexOfIndex = inst2.index(i);
	values[indexOfIndex] += inst2.valueSparse(i);
      }
      newInst = new SparseInstance(weight1+weight2, values);
      newInst.setDataset(m_Instances);
    }
    else if (!(inst1 instanceof SparseInstance) && !(inst2 instanceof SparseInstance)){
      for (int i=0; i<numAttributes; i++) {
	values[i] = inst1.value(i) + inst2.value(i);
      }
      newInst = new Instance(weight1+weight2, values);
      newInst.setDataset(m_Instances);
    }
    else {
      throw new Exception ("Error!! inst1 and inst2 should be both of same type -- sparse or non-sparse");
    }
    //      if (m_verbose) {
    //        System.out.println("Sum instance is: " + newInst);
    //      }
    return newInst;
  }

  /** Outputs the current clustering
   *
   * @exception Exception if something goes wrong
   */
  public void printIndexClusters() throws Exception {
    for (int j = 0; j < m_NumClusters; j++) {
      System.out.println("Cluster " + j);
      for (int i=0; i<m_Instances.numInstances(); i++) {
	System.out.println("Point: " + i + ", prob: " + m_ClusterDistribution[i][j]);
      }
    }
  }


  /** E-step of the KMeans clustering algorithm -- find new cluster
   assignments and new objective function 
  */
  protected double findAssignments() throws Exception{
    double m_Objective = 0;
    for (int i=0; i<m_Instances.numInstances(); i++) {
      Instance inst = m_Instances.instance(i);
      
      try {
	// Update cluster assignment probs
	m_Objective += assignInstanceToClustersWithConstraints(i);
      }
      catch (Exception e) {
	System.out.println("Could not find distance. Exception: " + e);
	e.printStackTrace();
      }
    }
    return m_Objective;
  }

  /**
   * Classifies the instance using the current clustering considering
   * constraints, updates cluster assignment probs
   *
   * @param instance the instance to be assigned to a cluster
   * @exception Exception if instance could not be assigned to clusters
   * successfully */

  public double assignInstanceToClustersWithConstraints(int instIdx) throws Exception {
    double objectiveForInstIdx = 0;
    for (int j = 0; j < m_NumClusters; j++) {
      if (!m_objFunDecreasing) {
	double sim = similarityInPottsModel(instIdx, j);
	m_ClusterDistribution[instIdx][j] = Math.exp(m_Kappa*sim); // similarity	
	//	System.out.println("Weight value for sim between instance " + instIdx + " and cluster " + j + " = " + m_ClusterDistribution[instIdx][j] + ", sim = " + sim);
      } else {
	double dist = squareDistanceInPottsModel(instIdx, j);
	m_ClusterDistribution[instIdx][j] = Math.exp(-m_Kappa*dist); // distance
	//	System.out.println("Weight value for dist between instance " + instIdx + " and cluster " + j + " = " + m_ClusterDistribution[instIdx][j] + ", dist = " + dist);
      }
    }
    //    System.out.println();
    
    if (!m_objFunDecreasing) {
      objectiveForInstIdx = Math.log(Utils.sum(m_ClusterDistribution[instIdx]));
    } else {
      objectiveForInstIdx = -Math.log(Utils.sum(m_ClusterDistribution[instIdx]));
    }

    // normalize to get posterior probs of cluster assignment
    Utils.normalize(m_ClusterDistribution[instIdx]);

    if (m_verbose) {
      System.out.println("Obj component is: " + objectiveForInstIdx);
      System.out.println("Posteriors for instance: " + instIdx);
      for (int j = 0; j < m_NumClusters; j++) {
	System.out.print(m_ClusterDistribution[instIdx][j] + " ");
      }
      System.out.println();
    }

    return objectiveForInstIdx;
  }

  /** finds similarity between instance and centroid in Potts Model with Relaxation Labeling
   */
  double similarityInPottsModel(int instIdx, int centroidIdx) throws Exception{
    double sim = m_metric.similarity(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));

    if (false) {
    if (m_ConstraintsHash != null) {
      HashMap cluster = m_IndexClusters[centroidIdx];
      if (cluster != null) {
	Iterator iter = cluster.entrySet().iterator();
	while(iter.hasNext()) {
	  int j = ((Integer) iter.next()).intValue();
	  int first = (j<instIdx)? j:instIdx;
	  int second = (j>=instIdx)? j:instIdx;
	  InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
	  if (m_ConstraintsHash.containsKey(pair)) {
	    int linkType = ((Integer) m_ConstraintsHash.get(pair)).intValue();
	    if (linkType == InstancePair.MUST_LINK) {  // count up number of must-links satisfied, instead of number of must-links violated. So, add to sim
	      if (m_verbose) {
		System.out.println("Found satisfied must link between: " + first + " and " + second);
	      }
	      sim += m_MustLinkWeight;
	    }
	    else if (linkType == InstancePair.CANNOT_LINK) {
	      if (m_verbose) {
		System.out.println("Found violated cannot link between: " + first + " and " + second);
	      }
	      sim -= m_CannotLinkWeight;
	    }
	  }
	}
      } // end while
    }
    }
    return sim;
  }

  /** finds squaredistance between instance and centroid in Potts Model with Relaxation Labeling
   */
  double squareDistanceInPottsModel(int instIdx, int centroidIdx) throws Exception{
    double dist = m_metric.distance(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));
    dist *= dist; // doing the squaring here itself
    if(m_verbose) {
      System.out.println("Unconstrained distance between instance " + instIdx + " and centroid " + centroidIdx + " is: " + dist);
    }

    if (false) {
    if (m_ConstraintsHash != null) {
      HashMap cluster = m_IndexClusters[centroidIdx];
      if (cluster != null) {
	Iterator iter = cluster.entrySet().iterator();
	while(iter.hasNext()) {
	  int j = ((Integer) iter.next()).intValue();
	  int first = (j < instIdx)? j:instIdx;
	  int second = (j >= instIdx)? j:instIdx;
	  InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
	  if (m_ConstraintsHash.containsKey(pair)) {
	    int linkType = ((Integer) m_ConstraintsHash.get(pair)).intValue();
	    if (linkType == InstancePair.MUST_LINK) { // count up number of must-links satisfied, instead of number of must-links violated. So, subtract from dist
	      if (m_verbose) {
		System.out.println("Found satisfied must link between: " + first + " and " + second);
	      }
	      dist -= m_MustLinkWeight;
	    }
	    else if (linkType == InstancePair.CANNOT_LINK) {
	      if (m_verbose) {
		System.out.println("Found violated cannot link between: " + first + " and " + second);
	      }
	      dist += m_CannotLinkWeight;
	    }
	  }
	}
      }
    }
    }
    if(m_verbose) {
      System.out.println("Final distance between instance " + instIdx + " and centroid " + centroidIdx + " is: " + dist);
    }
    return dist;
  }

  /** M-step of the KMeans clustering algorithm -- updates cluster centroids
   */
  protected void updateClusterCentroids() throws Exception {
    // M-step: update cluster centroids
    Instances [] tempI = new Instances[m_NumClusters];
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    
    for (int j = 0; j < m_NumClusters; j++) {
      tempI[j] = new Instances(m_Instances, 0); // tempI[j] stores the cluster instances for cluster j
    }
    for (int i = 0; i < m_Instances.numInstances(); i++) {
      for (int j = 0; j < m_NumClusters; j++) {
	tempI[j].add(m_Instances.instance(i), m_ClusterDistribution[i][j]); // instance weight holds the posterior prob of the instance in the cluster
      }
    }
    
    // Calculates cluster centroids
    for (int j = 0; j < m_NumClusters; j++) {
      double [] values = new double[m_Instances.numAttributes()];
      if (m_FastMode && m_isSparseInstance) {
	values = meanOrMode(tempI[j]); // uses fast meanOrMode
      }
      else {
	for (int k = 0; k < m_Instances.numAttributes(); k++) {
	  values[k] = tempI[j].meanOrMode(k); // uses usual meanOrMode
	}
      }
      
      // cluster centroids are dense in SPKMeans
      m_ClusterCentroids.add(new Instance(1.0, values));
      if (m_Algorithm == ALGORITHM_SPHERICAL) {
	try {
	  normalize(m_ClusterCentroids.instance(j));
	}
	catch (Exception e) {
	  e.printStackTrace();
	}
      }
    }
    for (int j = 0; j < m_NumClusters; j++) {
      tempI[j] = null; // free memory for garbage collector to pick up
    }
  }
  
  
  /** Actual KMeans function */
  protected void runEM() throws Exception {
    boolean converged = false;
    m_Iterations = 0;

    m_ClusterDistribution = new double [m_Instances.numInstances()][m_NumClusters];
    
    // initialize cluster distribution from the cluster assignments
    // after initial neighborhoods have been built
    for (int i=0; i<m_Instances.numInstances(); i++) {
      for (int j=0; j<m_NumClusters; j++) {
	m_ClusterDistribution[i][j] = 0;
      }
      if (m_ClusterAssignments[i] != -1 && m_ClusterAssignments[i] < m_NumClusters) {
	m_ClusterDistribution[i][m_ClusterAssignments[i]] = 1;
      }
    }

    double oldObjective = m_objFunDecreasing ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;

    while (!converged) {
      // E-step: updates m_Objective
      if (m_verbose) {
	System.out.println("Doing E-step ...");
      }

      m_Objective = findAssignments(); // finds assignments and calculates objective function

      // M-step
      if (m_verbose) {
	System.out.println("Doing M-step ...");
      }
      updateClusterCentroids(); 
      m_Iterations++;

      // anneal the value of kappa
      if (!m_objFunDecreasing) {
	if (m_Kappa < m_MaxKappaSim) {
	  m_Kappa *= 2;
	}
      } else {
	if (m_Kappa < m_MaxKappaDist) {
	  m_Kappa += 2;
	}
      }
      
      // Convergence check
      if(Math.abs(oldObjective - m_Objective) > m_ObjFunConvergenceDifference) {
	System.out.println("Objective function: " + m_Objective + ", numIterations = " + m_Iterations);
	converged = false;
      }
      else {
	converged = true;
	System.out.println("Final Objective function is: " + m_Objective + ", numIterations = " + m_Iterations);
      }
      if ((!m_objFunDecreasing && oldObjective > m_Objective) ||
  	  (m_objFunDecreasing && oldObjective < m_Objective)) {
	//	throw new Exception("Oscillations => bug in objective function/EM step!!");
      }
      oldObjective = m_Objective;
    }
  }

  /** Dummy: not implemented for PCSoftKMeans */
  public int[] bestInstancesForActiveLearning(int numActive) throws Exception{
    throw new Exception("Not implemented for PCSoftKMeans");
  }

  /** Dummy: not implemented for PCSoftKMeans */
  public InstancePair[] bestPairsForActiveLearning(int numActive) throws Exception{
    throw new Exception("Not implemented for PCSoftKMeans");
  }


  /**
   * Checks if instance has to be normalized and returns the
   * distribution of the instance using the current clustering
   *
   * @param instance the instance under consideration
   * @return an array containing the estimated membership 
   * probabilities of the test instance in each cluster (this 
   * should sum to at most 1)
   * @exception Exception if distribution could not be 
   * computed successfully
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    if (m_Algorithm == ALGORITHM_SPHERICAL) { // check here, since evaluateModel calls this function on test data
      normalize(instance);
    }
    return null;
  }

  /**
   * Computes the density for a given instance.
   * 
   * @param inst the instance to compute the density for
   * @return the density.
   * @exception Exception if the density could not be computed
   * successfully
   */
  public double densityForInstance(Instance inst) throws Exception {
    throw new Exception("Not implemented for PCSoftKMeans, since posterior probs directly computed and density weights not stored!!");
  }


  /** lookup the instance in the checksum hash
   * @param instance instance to be looked up
   * @return the index of the cluster to which the instance was assigned, -1 if the instance has not bee clustered
   */
  protected int lookupInstanceCluster(Instance instance) {
    int classIdx = instance.classIndex();
    double[] values1 = instance.toDoubleArray();
    double checksum = 0; 
    for (int i = 0; i < values1.length; i++) {
      if (i != classIdx) {
	checksum += m_checksumCoeffs[i] * values1[i]; 
      } 
    }

    Object list = m_checksumHash.get(new Double(checksum));
    if (list != null) {
      // go through the list of instances with the same checksum and find the one that is equivalent
      ArrayList checksumList = (ArrayList) list;
      for (int i = 0; i < checksumList.size(); i++) {
	int instanceIdx = ((Integer) checksumList.get(i)).intValue();
	Instance listInstance = m_Instances.instance(instanceIdx);
	double[] values2 = listInstance.toDoubleArray();
	boolean equal = true; 
	for (int j = 0; j < values1.length && equal == true; j++) {
	  if (j != classIdx) {
	    if (values1[j] != values2[j]) {
	      equal = false;
	    }
	  } 
	}
	if (equal == true) {
	  return m_ClusterAssignments[instanceIdx]; 
	}
      } 
    } 
    return -1; 
  }

  /**
   * Classifies the instance using the current clustering, without considering constraints
   *
   * @param instance the instance to be assigned to a cluster
   * @return the number of the assigned cluster as an integer
   * if the class is enumerated, otherwise the predicted value
   * @exception Exception if instance could not be classified
   * successfully */

  public int assignInstanceToCluster(Instance instance) throws Exception {
    int bestCluster = 0;
    double bestDistance = Double.POSITIVE_INFINITY;
    double bestSimilarity = Double.NEGATIVE_INFINITY;

    // lookup the cluster assignment of the instance
    int lookupCluster = lookupInstanceCluster(instance);
    if (lookupCluster >= 0) {
      return lookupCluster;
    }

    System.err.println("\n\nWARNING!!! Couldn't lookup instance - it wasn't in the clustering!!!\n\n");

    System.err.println("\n\n Assuming user wants prediction of new test instance based on clustering model ...\nDoing instance assignment without constraints!!!\n\n");
    for (int i = 0; i < m_NumClusters; i++) {
      double distance = 0, similarity = 0;
      if (!m_objFunDecreasing) {
	similarity = m_metric.similarity(instance, m_ClusterCentroids.instance(i));
	if (similarity > bestSimilarity) {
	  bestSimilarity = similarity;
	  bestCluster = i;
	}
      } else {
	distance = m_metric.distance(instance, m_ClusterCentroids.instance(i));
	if (distance < bestDistance) {
	  bestDistance = distance;
	  bestCluster = i;
	}
      }
    }
    return bestCluster;
  }
  
  /** Set the cannot link constraint weight */
  public void setCannotLinkWeight(double w) {
    m_CannotLinkWeight = w;
  }

  /** Return the cannot link constraint weight */
  public double getCannotLinkWeight() {
    return m_CannotLinkWeight;
  }

  /** Set the must link constraint weight */
  public void setMustLinkWeight(double w) {
    m_MustLinkWeight = w;
  }

  /** Return the must link constraint weight */
  public double getMustLinkWeight() {
    return m_MustLinkWeight;
  }

  /** Return the number of clusters */
  public int getNumClusters() {
    return m_NumClusters;
  }

  /** A duplicate function to conform to Clusterer abstract class.
   * @returns the number of clusters
   */
  public int numberOfClusters() {
    return getNumClusters();
  } 
  
  /** Set the m_SeedHash */
  public void setSeedHash(HashMap seedhash) {
    System.err.println("Not implemented here");
  }    

 /**
   * Set the random number seed
   * @param s the seed
   */
  public void setRandomSeed (int s) {
    m_RandomSeed = s;
  }

    
  /** Return the random number seed */
  public int getRandomSeed () {
    return  m_RandomSeed;
  }

  /**
   * Set the minimum value of the objective function difference required for convergence
   * @param objFunConvergenceDifference the minimum value of the objective function difference required for convergence
   */
  public void setObjFunConvergenceDifference(double objFunConvergenceDifference) {
    m_ObjFunConvergenceDifference = objFunConvergenceDifference;
  }

    /**
   * Get the minimum value of the objective function difference required for convergence
   * @returns the minimum value of the objective function difference required for convergence
   */
  public double getObjFunConvergenceDifference() {
    return m_ObjFunConvergenceDifference;
  }
    
 /** Sets training instances */
  public void setInstances(Instances instances) {
    m_Instances = instances;

    // create the checksum coefficients
    m_checksumCoeffs = new double[instances.numAttributes()];
    Random r = new Random(instances.numInstances()); 
    for (int i = 0; i < m_checksumCoeffs.length; i++) {
      m_checksumCoeffs[i] = r.nextDouble();
    }

    // hash the instance checksums
    m_checksumHash = new HashMap(instances.numInstances());
    int classIdx = instances.classIndex();
    for (int i = 0; i < instances.numInstances(); i++) {
      Instance instance = instances.instance(i);
      double[] values = instance.toDoubleArray();
      double checksum = 0;

      for (int j = 0; j < values.length; j++) {
	if (j != classIdx) {
	  checksum += m_checksumCoeffs[j] * values[j]; 
	} 
      }

      // take care of chaining
      Object list = m_checksumHash.get(new Double(checksum));
      ArrayList idxList = null; 
      if (list == null) {
	idxList = new ArrayList();
	m_checksumHash.put(new Double(checksum), idxList);
      } else { // chaining
	idxList = (ArrayList) list;
      }
      idxList.add(new Integer(i));
    } 
  }


  /** Return training instances */
  public Instances getInstances() {
    return m_Instances;
  }

  /**
   * Set the number of clusters to generate
   *
   * @param n the number of clusters to generate
   */
  public void setNumClusters(int n) {
    m_NumClusters = n;
    if (m_verbose) {
      System.out.println("Number of clusters: " + n);
    }
  }


  /**
   * Set the distance metric
   *
   * @param s the metric
   */
  public void setMetric (LearnableMetric m) {
    m_metric = m;
    String metricName = m_metric.getClass().getName();
    System.out.println("Setting m_metric to " + metricName);
    m_objFunDecreasing = m.isDistanceBased();
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
   * Set the KMeans algorithm.  Values other than
   * ALGORITHM_SIMPLE or ALGORITHM_SPHERICAL will be ignored
   *
   * @param algo algorithm type
   */
  public void setAlgorithm (SelectedTag algo)
  {
    if (algo.getTags() == TAGS_ALGORITHM) {
      if (m_verbose) {
	System.out.println("Algorithm: " + algo.getSelectedTag().getReadable());
      }
      m_Algorithm = algo.getSelectedTag().getID();
    }
  }

  /**
   * Get the KMeans algorithm type. Will be one of
   * ALGORITHM_SIMPLE or ALGORITHM_SPHERICAL
   *
   * @returns algorithm type
   */
  public SelectedTag getAlgorithm ()
  {
    return new SelectedTag(m_Algorithm, TAGS_ALGORITHM);
  }

    
  /** Read the seeds from a hastable, where every key is an instance and every value is:
   * the cluster assignment of that instance 
   * seedVector vector containing seeds
   */
  
  public void seedClusterer(HashMap seedHash) {
    System.err.println("Not implemented here");
  }
 

  /** Prints clusters */
  public void printClusters () throws Exception{
    ArrayList clusters = getClusters();
    for (int i=0; i<clusters.size(); i++) {
      Cluster currentCluster = (Cluster) clusters.get(i);
      System.out.println("\nCluster " + i + ": " + currentCluster.size() + " instances");
      if (currentCluster == null) {
	System.out.println("(empty)");
      }
      else {
	for (int j=0; j<currentCluster.size(); j++) {
	  Instance instance = (Instance) currentCluster.get(j);	
	  System.out.println("Instance: " + instance);
	}
      }
    }
  }

  /**
   * Computes the clusters from the cluster assignments, for external access
   * 
   * @exception Exception if clusters could not be computed successfully
   */    

  public ArrayList getClusters() throws Exception {
    m_Clusters = new ArrayList();
    Cluster [] clusterArray = new Cluster[m_NumClusters];

    for (int i=0; i < m_Instances.numInstances(); i++) {
	Instance inst = m_Instances.instance(i);
	if(clusterArray[m_ClusterAssignments[i]] == null)
	   clusterArray[m_ClusterAssignments[i]] = new Cluster();
	clusterArray[m_ClusterAssignments[i]].add(inst, 1);
    }

    for (int j =0; j< m_NumClusters; j++) 
      m_Clusters.add(clusterArray[j]);

    return m_Clusters;
  }

  /**
   * Computes the clusters from the cluster assignments, for external access
   * 
   * @exception Exception if clusters could not be computed successfully
   */    

  public HashMap[] getIndexClusters() throws Exception {
//      m_IndexClusters = new HashMap[m_NumClusters];
//      for (int j=0; j < m_NumClusters; j++) { 
//        m_IndexClusters[j] = new HashMap();
//        for (int i=0; i < m_Instances.numInstances(); i++) {
//  	m_IndexClusters[j].put(new Integer(i),new Double(m_ClusterDistribution[i][j]));
//        }
//      }
//      return m_IndexClusters;
    return null;
  }


  public Enumeration listOptions () {
    Vector newVector = new Vector(10);

     newVector.addElement(new Option("\tnumber of clusters (default = 3)." 
				    , "N", 1, "-N <num>"));
     newVector.addElement(new Option("\trandom number seed (default 1)"
				     , "R", 1, "-R <num>"));
     newVector.addElement(new Option("\tperform no seeding (default false)"
				     , "NS", 1, "-NS"));
     newVector.addElement(new Option("\tperform active learning (default false)"
				     , "A", 1, "-A"));
     newVector.addElement(new Option("\tphase two of active learning is random (default false)"
				     , "P2", 1, "-P2"));
     newVector.addElement(new Option("\tdo only Explore phase in active learning (default false)"
				     , "E", 1, "-E"));
     newVector.addElement(new Option("\tmetric type (default WeightedEuclidean)"
				     , "M", 1, "-M <string> (WeightedEuclidean or WeightedDotP)"));     
     newVector.addElement(new Option("\tconstraints file"
				     , "C", 1, "-C <string> (each line is of the form \"firstID\\tsecondID\\t<+1/-1>\", where +1=>must-link, -1=>cannot-link)"));
     newVector.addElement(new Option("\tmust link weight (default 1)"
				     , "ML", 1, "-ML <double>"));
     newVector.addElement(new Option("\tcannot link weight (default 1)"
				     , "CL", 1, "-CL <double>"));
     newVector.addElement(new Option("\talgorithm type (default Simple)"
				     , "A", 1, "-A <string> (Simple => Simple-KMeans, Spherical => Spherical-KMeans)"));

     return  newVector.elements();

  }

  public String [] getOptions ()  {
    String[] options = new String[80];
    int current = 0;
    
    if (!m_Seedable) {
      options[current++] = "-NS";
    }
    
    options[current++] = "-A";
    options[current++] = "" + getAlgorithm().getSelectedTag().getID();
    options[current++] = "-N";
    options[current++] = "" + getNumClusters();
    options[current++] = "-R";
    options[current++] = "" + getRandomSeed();
    options[current++] = "-ML";
    options[current++] = "" + m_MustLinkWeight;
    options[current++] = "-CL";
    options[current++] = "" + m_CannotLinkWeight;
    
    options[current++] = "-M";
    options[current++] = Utils.removeSubstring(m_metric.getClass().getName(), "weka.core.metrics.");
    if (m_metric instanceof OptionHandler) {
      String[] metricOptions = ((OptionHandler)m_metric).getOptions();
      for (int i = 0; i < metricOptions.length; i++) {
	options[current++] = metricOptions[i];
      }
    } 
    
    while (current < options.length) {
      options[current++] = "";
    }
    
    return  options;
  }
  
  /**
   * Parses a given list of options.
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   *
   **/
  public void setOptions (String[] options)
    throws Exception {

    String optionString = Utils.getOption('N', options);
    if (optionString.length() != 0) {
      setNumClusters(Integer.parseInt(optionString));
    }

    optionString = Utils.getOption('R', options);
    if (optionString.length() != 0) {
      setRandomSeed(Integer.parseInt(optionString));
    }

    optionString = Utils.getOption('A', options);
    if (optionString.length() != 0) {
      setAlgorithm(new SelectedTag(Integer.parseInt(optionString), TAGS_ALGORITHM));
    }

    optionString = Utils.getOption('M', options);
    if (optionString.length() != 0) {
      String[] metricSpec = Utils.splitOptions(optionString);
      String metricName = metricSpec[0]; 
      metricSpec[0] = "";
      setMetric((LearnableMetric) LearnableMetric.forName(metricName, metricSpec));
    }
  }

  /**   
   * return a string describing this clusterer
   *
   * @return a description of the clusterer as a string
   */
  public String toString() {
    StringBuffer temp = new StringBuffer();

    temp.append("\nkMeans\n======\n");
    temp.append("\nNumber of iterations: " + m_Iterations+"\n");

//      temp.append("\nCluster centroids:\n");
//      for (int i = 0; i < m_NumClusters; i++) {
//        temp.append("\nCluster "+i+"\n\t");
//      }
//      temp.append("\n");
    return temp.toString();
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

  /** Normalizes Instance or SparseInstance
   *
   * @author Sugato Basu
   * @param inst Instance to be normalized
   */

  public void normalize(Instance inst) throws Exception {
    if (inst instanceof SparseInstance) {
      normalizeSparseInstance(inst);
    }
    else {
      normalizeInstance(inst);
    }
  }

  /** Normalizes the values of a normal Instance in L2 norm
   *
   * @author Sugato Basu
   * @param inst Instance to be normalized
   */

  public void normalizeInstance(Instance inst) throws Exception{
    double norm = 0;
    double values [] = inst.toDoubleArray();

    if (inst instanceof SparseInstance) {
      System.err.println("Is SparseInstance, using normalizeSparseInstance function instead");
      normalizeSparseInstance(inst);
    }
    
    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	norm += values[i] * values[i];
      }
    }
    norm = Math.sqrt(norm);
    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	values[i] /= norm;
      }
    }
    inst.setValueArray(values);
  }

  /** Normalizes the values of a SparseInstance in L2 norm
   *
   * @author Sugato Basu
   * @param inst SparseInstance to be normalized
   */

  public void normalizeSparseInstance(Instance inst) throws Exception{
    double norm=0;
    int length = inst.numValues();

    if (!(inst instanceof SparseInstance)) {
      System.err.println("Not SparseInstance, using normalizeInstance function instead");
      normalizeInstance(inst);
    }

    for (int i=0; i<length; i++) {
      if (inst.index(i) != inst.classIndex()) { // don't normalize the class index
	norm += inst.valueSparse(i) * inst.valueSparse(i);
      }
    }
    norm = Math.sqrt(norm);
    for (int i=0; i<length; i++) { // don't normalize the class index
      if (inst.index(i) != inst.classIndex()) {
	inst.setValueSparse(i, inst.valueSparse(i)/norm);
      }
    }
  }
  
  /** Fast version of meanOrMode - streamlined from Instances.meanOrMode for efficiency 
   *  Does not check for missing attributes, assumes numeric attributes, assumes Sparse instances
   */

  protected double[] meanOrMode(Instances insts) {

    int numAttributes = insts.numAttributes();
    double [] value = new double[numAttributes];
    double weight = 0;
    
    for (int i=0; i<numAttributes; i++) {
      value[i] = 0;
    }

    for (int j=0; j<insts.numInstances(); j++) {
      SparseInstance inst = (SparseInstance) (insts.instance(j));
      weight += inst.weight();
      for (int i=0; i<inst.numValues(); i++) {
	int indexOfIndex = inst.index(i);
	value[indexOfIndex]  += inst.weight() * inst.valueSparse(i);
      }
    }
    
    if (Utils.eq(weight, 0)) {
      for (int k=0; k<numAttributes; k++) {
	value[k] = 0;
      }
    }
    else {
      for (int k=0; k<numAttributes; k++) {
	value[k] = value[k] / weight;
      }
    }

    return value;
  }

  /**
   * Gets a Double representing the current date and time.
   * eg: 1:46pm on 20/5/1999 -> 19990520.1346
   *
   * @return a value of type Double
   */
  public static Double getTimeStamp() {

    Calendar now = Calendar.getInstance(TimeZone.getTimeZone("UTC"));
    double timestamp = now.getTimeInMillis();
    return new Double(timestamp);
  }


  /**
   * Main method for testing this class.
   *
   */

  public static void main (String[] args) {
    try {    
      testCase();
      //System.out.println(ClusterEvaluation.evaluateClusterer(new PCSoftKMeans(), args));
    }
    catch (Exception e) {
      System.out.println(e.getMessage());
      e.printStackTrace();
    }
  }

  protected static void testCase() {
    try {
      //String dataset = new String("lowd");
      String dataset = new String("highd");
      if (dataset.equals("lowd")) {
	//////// Low-D data
	String datafile = "/u/ml/software/weka-latest/data/iris.arff";
	
	// set up the data
	FileReader reader = new FileReader (datafile);
	Instances data = new Instances (reader);
	
	// Make the last attribute be the class 
	int classIndex = data.numAttributes()-1;
	data.setClassIndex(classIndex); // starts with 0
	System.out.println("ClassIndex is: " + classIndex);
	
	// Remove the class labels before clustering
	Instances clusterData = new Instances(data);
	clusterData.deleteClassAttribute();
		
	// create random constraints from the labeled training data
	int numPairs = 10, num=0;
	ArrayList labeledPair = new ArrayList(numPairs);
	Random rand = new Random(42);
	System.out.println("Initializing constraint matrix:");
	while (num < numPairs) {
	  int i = (int) (data.numInstances()*rand.nextFloat());	
	  int j = (int) (data.numInstances()*rand.nextFloat());
	  int first = (i<j)? i:j;
	  int second = (i>=j)? i:j;
	  int linkType = (data.instance(first).classValue() == 
			  data.instance(second).classValue())? 
	    InstancePair.MUST_LINK:InstancePair.CANNOT_LINK;
	  InstancePair pair = new InstancePair(first, second, linkType);
	  if (first!=second && !labeledPair.contains(pair)) {
	    labeledPair.add(pair);
	    num++;
	  }
	}
	System.out.println("Finished initializing constraints");
	
	// create clusterer
	PCSoftKMeans pckmeans = new PCSoftKMeans();
	System.out.println("\nClustering the iris data using PCKmeans...\n");
	pckmeans.setAlgorithm(new SelectedTag(ALGORITHM_SIMPLE, TAGS_ALGORITHM));
	WeightedEuclidean euclidean = new WeightedEuclidean();
	euclidean.setExternal(false);
	pckmeans.setMetric(euclidean);
	pckmeans.setVerbose(false);
	pckmeans.setSeedable(false);
	pckmeans.setNumClusters(data.numClasses());

	// do clustering
	pckmeans.buildClusterer(labeledPair, clusterData, data, data.numInstances());
	pckmeans.printIndexClusters();
      }
      else if (dataset.equals("highd")) {
	//////// Newsgroup data
	String datafile = "/u/ml/data/CCSfiles/arffFromCCS/different-100_fromCCS.arff";
	
	// set up the data
	FileReader reader = new FileReader (datafile);
	Instances data = new Instances (reader);
	
	// Make the last attribute be the class 
	int classIndex = data.numAttributes()-1;
	data.setClassIndex(classIndex); // starts with 0
	System.out.println("ClassIndex is: " + classIndex);
	
	// Remove the class labels before clustering
	Instances clusterData = new Instances(data);
	clusterData.deleteClassAttribute();
	
	// create random constraints from the labeled training data
	int numPairs = 100, num=0;
	ArrayList labeledPair = new ArrayList(numPairs);
	Random rand = new Random(42); 
	System.out.println("Initializing constraint matrix:");
	while (num < numPairs) {
	  int i = (int) (data.numInstances()*rand.nextFloat());	
	  int j = (int) (data.numInstances()*rand.nextFloat());
	  int first = (i<j)? i:j;
	  int second = (i>=j)? i:j;
	  int linkType = (data.instance(first).classValue() == 
			  data.instance(second).classValue())? 
	    InstancePair.MUST_LINK:InstancePair.CANNOT_LINK;
	  InstancePair pair = new InstancePair(first, second, linkType);
	  if (first!=second && !labeledPair.contains(pair)) {
	    labeledPair.add(pair);
	    num++;
	  }
	}
	System.out.println("Finished initializing constraints");
	
	// create clusterer
	PCSoftKMeans pckmeans = new PCSoftKMeans();
	System.out.println("\nClustering the news data using PCKmeans...\n");
	pckmeans.resetClusterer();
	pckmeans.setAlgorithm(new SelectedTag(ALGORITHM_SPHERICAL, TAGS_ALGORITHM));
	WeightedDotP dotp = new WeightedDotP();
	dotp.setExternal(false);
	dotp.setLengthNormalized(true);
	pckmeans.setMetric(dotp);
	pckmeans.setVerbose(false);
	pckmeans.setSeedable(true);
	pckmeans.setNumClusters(data.numClasses());

	// do clustering
	pckmeans.buildClusterer(labeledPair, clusterData, data, clusterData.numInstances());
	pckmeans.printIndexClusters();
      }
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }
}

// TODO: Add init using farthest first
