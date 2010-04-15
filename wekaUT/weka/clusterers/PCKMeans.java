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
 *    PCKMeans.java
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
public class PCKMeans extends Clusterer implements OptionHandler,SemiSupClusterer,ActiveLearningClusterer {

  /** Name of clusterer */
  String m_name = "PCKMeans";

  /** holds the instances in the clusters */
  protected ArrayList m_Clusters = null;

  /** holds the instance indices in the clusters */
  protected HashSet[] m_IndexClusters = null;
  
  /** holds the ([instance pair] -> [type of constraint])
      mapping. Note that the instance pairs stored in the hash always
      have constraint type InstancePair.DONT_CARE_LINK, the actual
      link type is stored in the hashed value 
  */
  
  protected HashMap m_ConstraintsHash = null;
  
  /** holds the ([instance i] -> [Arraylist of constraints involving i])
      mapping. Note that the instance pairs stored in the Arraylist 
      have the actual link type
  */
  protected HashMap m_instanceConstraintHash = null; 

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

  /** Round robin or Random in active Phase Two */
  protected boolean m_PhaseTwoRandom = false;

  /** Two-phase active learning or All Explore */
  protected boolean m_AllExplore = false;

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
  
  /** min difference of objective function values for convergence*/
  protected double m_ObjFunConvergenceDifference = 1e-5;

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
   * number of pairs to seed with
   */
  protected int m_NumActive;

  /**
   * active mode?
   */
  protected boolean m_Active = false;

  /**
   * number of clusters to generate, default is -1 to get it from labeled data
   */
  protected int m_NumClusters = 3;


  /** Number of clusters in the process*/
  protected int m_NumCurrentClusters = 0; 


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
   * temporary variable holding cluster assignments while iterating
   */
  protected int [] m_ClusterAssignments;

  /**
   * temporary variable holding cluster sums while iterating
   */
  protected Instance [] m_SumOfClusterInstances;

  /**
   * holds the random Seed used to seed the random number generator
   */
  protected int m_RandomSeed = 42;

  /**
   * holds the random number generator used in various parts of the code
   */
  protected Random m_RandomNumberGenerator = null;

  /** Define possible orderings */
  public static final int ORDERING_DEFAULT = 1;
  public static final int ORDERING_RANDOM = 2;
  public static final int ORDERING_SORTED = 3;
  public static final Tag[] TAGS_ORDERING = {
    new Tag(ORDERING_DEFAULT, "Default-Ordering"),
    new Tag(ORDERING_RANDOM, "Random-Ordering"),
    new Tag(ORDERING_SORTED, "Sorted-Ordering")
      };

  protected int m_InstanceOrdering = ORDERING_DEFAULT;

  /** Move points in assignment step till stabilization? */
  protected boolean m_MovePointsTillAssignmentStabilizes = false;

  /** neighbor list for active learning: points in each cluster neighborhood */
  protected HashSet[] m_NeighborSets;

  /** assigned set for active learning: whether a point has been assigned or not */
  HashSet m_AssignedSet;

  /* Constructor */
  public PCKMeans() {
  }

  /* Constructor */
  public PCKMeans(Metric metric) {
    m_metric = metric;
    m_objFunDecreasing = metric.isDistanceBased();
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
   * Generates the clustering using labeled seeds
   *
   * @param labeledData set of labeled instances to use as seeds
   * @param unlabeledData set of unlabeled instances
   * @param classIndex attribute index in labeledData which holds class info
   * @param numClusters number of clusters to create
   * @param startingIndexOfTest from where test data starts in unlabeledData, useful if clustering is transductive, set to -1 if not relevant
   * @exception Exception if something is wrong
   */
  public void buildClusterer (Instances labeledData, Instances unlabeledData, int classIndex, int numClusters, int startingIndexOfTest) throws Exception {
    // Dummy function
    throw new Exception ("Not implemented for MPCKMeans, only here for compatibility to SemiSupClusterer interface");
  }


  /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using labeledData as seeds
   *
   * @param labeledPairs labeled instances to be used as seeds
   * @param unlabeledData unlabeled training (+ test for transductive) instances
   * @param labeledTrain labeled training instances
   * @param startingIndexOfTest starting index of test set in unlabeled data
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(ArrayList labeledPairs, Instances unlabeledData, Instances labeledTrain,
			     int startingIndexOfTest) throws Exception {
    int classIndex = labeledTrain.numAttributes(); // assuming that the last attribute is always the class
    m_TotalTrainWithLabels = labeledTrain;

    m_SeedHash = new HashSet((int) (unlabeledData.numInstances()/0.75 + 10)) ;
    m_ConstraintsHash = new HashMap(m_MaxConstraintsAllowed); 
    m_instanceConstraintHash = new HashMap(m_MaxConstraintsAllowed);

    if (!m_Active && labeledPairs != null) {
      for (int i=0; i<labeledPairs.size(); i++) {
	InstancePair pair = (InstancePair) labeledPairs.get(i);
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
	  // hash the constraints for the instances involved
	  Object constraintList1 = m_instanceConstraintHash.get(firstInt);
	  if (constraintList1 == null) {
	    ArrayList constraintList = new ArrayList();
	    constraintList.add(pair);
	    m_instanceConstraintHash.put(firstInt, constraintList);
	  } else {
	    ((ArrayList)constraintList1).add(pair);
	  }
	  Object constraintList2 = m_instanceConstraintHash.get(secondInt);
	  if (constraintList2 == null) {
	    ArrayList constraintList = new ArrayList();
	    constraintList.add(pair);
	    m_instanceConstraintHash.put(secondInt, constraintList);
	  } else {
	    ((ArrayList)constraintList2).add(pair);
	  }
	}
      }
    } else {
      m_NumActive = labeledPairs.size();
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
   * Generates a clusterer. Instances in data have to be
   * either all sparse or all non-sparse
   *
   * @param data set of instances serving as training data 
   * @exception Exception if the clusterer has not been 
   * generated successfully
   */
  public void buildClusterer(Instances data) throws Exception {
    System.out.println("Must link weight: " + m_MustLinkWeight);
    System.out.println("Cannot link weight: " + m_CannotLinkWeight);

    m_RandomNumberGenerator = new Random(m_RandomSeed);
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

    System.out.println("Initializing clustering ... ");
    
    if (m_Active) {
      bestPairsForActiveLearning(m_NumActive);
    }
    else {
      nonActivePairwiseInit();
    }
    System.out.println("Done initializing clustering ...");
    
    if (m_verbose) {
      if (m_Seedable) {
	System.out.println("Initial assignments of seed points:");
	getIndexClusters();
	printIndexClusters();
      }
      for (int i=0; i<m_NumClusters; i++) {
	System.out.println("Centroid " + i + ": " + m_ClusterCentroids.instance(i));
      }
    }
    runKMeans();
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
    m_instanceConstraintHash = null;
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

  /** Phase 1 code for active learning 
   */
  protected int activePhaseOne(int numQueries) throws Exception {
    int numInstances = m_Instances.numInstances();
    int X, Y, Z;
    int query, lambda, Label, CLcount;
    boolean MLmode = true;
    
    System.out.println("In Explore phase, with numqueries: " + numQueries);

    // these are the main data-structures to be updated here
    m_NeighborSets = new HashSet[m_NumClusters]; // set of points in each cluster neighborhood
    m_SumOfClusterInstances = new Instance[m_NumClusters];
    m_ClusterAssignments = new int[numInstances];
    m_AssignedSet = new HashSet((int) (numQueries/0.75+10)); // whether a point has been assigned or not

    for (int i=0; i<m_Instances.numInstances(); i++) {
      m_ClusterAssignments[i] = -1;
    }
    query = 0; // current num queries
    lambda = -1; // curent number of disjoint neighborhoods
    X = 0; // current point under investigation
    
    while( query < numQueries ){
      if( m_NeighborSets[0] == null ){
	// start the first neighborhood from the first point
	lambda++;
	if (m_verbose) 
	  System.out.println("Setting cluster of " + X + " to " + lambda);
	// update data structures
	m_NeighborSets[lambda] = new HashSet();
	m_NeighborSets[lambda].add(new Integer(X));
	m_SumOfClusterInstances[lambda] = sumWithInstance(m_SumOfClusterInstances[lambda],m_Instances.instance(X));
	m_ClusterAssignments[X] = lambda;
	m_AssignedSet.add(new Integer(X));
      }
      else if( lambda == m_NumClusters-1 && !m_AllExplore) {
	// NOTE: this condition is fired only if we are doing 2 phase (Explore + Consolidate)
	System.out.println("Explore phase over after " + query + " queries");
	m_NumCurrentClusters = lambda+1;
	return query;
      }
      else {
	Z = (int) farthestFromSet(m_AssignedSet, null);
	CLcount = -1;
	for( int h = 0; h <= lambda; h++ ){ 
	  if (m_verbose) 
	    System.out.println("Starting for loop CLcount: " + CLcount);
	  Iterator NbrIt = null;
	  if (m_NeighborSets[h] != null) {
	    NbrIt = m_NeighborSets[h].iterator();
	  }
	  if( NbrIt != null && NbrIt.hasNext() ){
	    X = ((Integer) NbrIt.next()).intValue();
	    if (m_verbose) 
	      System.out.println("Inside iterator next ... X: " + X);
	    
	    Label = askOracle(X,Z);
	    query++;
	    System.out.println("Making query: " + query);
	    if( Label == InstancePair.CANNOT_LINK ){ // Cannot-link, update CLcount
	      CLcount++;
	    }
	    else{  // Must-link, add to neighborset
	      // update data structures	      
	      m_NeighborSets[h].add(new Integer(Z));
	      m_SumOfClusterInstances[h] = sumWithInstance(m_SumOfClusterInstances[h],m_Instances.instance(Z));
	      m_ClusterAssignments[Z] = h;
	      m_AssignedSet.add(new Integer(Z));
	      break; // get out of for loop
	    }
	    if(query >= numQueries){
	      if (m_verbose) 
		System.out.println("Run out of queries");
	      m_NumCurrentClusters = lambda+1;
	      return query;
	    }
	  }
	}
	if (m_verbose) {
	  System.out.println("Out of for loop");
	}
	if( CLcount == lambda ){ // found a point cannot-linked to all current clusters
	  lambda++;
	  // update data structures
	  m_NeighborSets[lambda] = new HashSet();
	  m_NeighborSets[lambda].add(new Integer(Z));
	  m_SumOfClusterInstances[lambda] = sumWithInstance(m_SumOfClusterInstances[lambda],m_Instances.instance(Z));
	  m_ClusterAssignments[Z] = lambda;
	  m_AssignedSet.add(new Integer(Z));
	}
      } // close else
    } // close while

    if (m_verbose) 
      System.out.println("Number of queries: " + query);
    m_NumCurrentClusters = lambda+1;
    return query;
  }

  /** Phase 2 code for active learning, with round robin 
   */
  protected void activePhaseTwoRoundRobin(int numQueries) throws Exception {
    int numInstances = m_Instances.numInstances();
    int X,Y;
    int query = 0, Label;

    System.out.println("In Consolidate phase, with numqueries: " + numQueries);
    while( query < numQueries ){
      if (m_verbose)
	System.out.println("Starting round robin");
      
      // starting round robin
      Instance[] clusterCentroids = new Instance[m_NumClusters];
      
      // find cluster with smallest size
      int smallestSize = Integer.MAX_VALUE, smallestCluster = -1;
      for (int i=0; i<m_NumClusters; i++) {
	if (m_NeighborSets[i].size() < smallestSize) {
	  smallestSize = m_NeighborSets[i].size();
	  smallestCluster = i;
	}
      }
      
      // compute centroid for smallest cluster
      if (m_isSparseInstance) {
	clusterCentroids[smallestCluster] = new SparseInstance(m_SumOfClusterInstances[smallestCluster]);
      }
      else {
	clusterCentroids[smallestCluster] = new Instance(m_SumOfClusterInstances[smallestCluster]);
      }
      clusterCentroids[smallestCluster].setDataset(m_Instances);
      if (!m_objFunDecreasing) {
	normalize(clusterCentroids[smallestCluster]);
      }
      else {
	normalizeByWeight(clusterCentroids[smallestCluster]);
      }
      
      // find next point, closest to centroid of smallest cluster
      X = nearestFromPoint(clusterCentroids[smallestCluster], m_AssignedSet);
      if (X == -1) {
	if (m_verbose)
	  System.out.println("No more points left unassigned, we are DONE!!");
	createGlobalCentroids();
	addMLAndCLTransitiveClosure(null);
	return;
      }
      
      if (m_verbose)
	System.out.println("Nearest point is " + X);
      if (X >= m_StartingIndexOfTest) { // Sanity Check
	throw new Exception ("Test point selected, something went wrong!");
      }      
      
      Iterator NbrIt = m_NeighborSets[smallestCluster].iterator();
      Y = ((Integer) NbrIt.next()).intValue(); // get any point from the smallest neighborhood
      Label = askOracle(X,Y);
      query++;
      System.out.println("Making query:" + query);
      if (m_verbose) 
	System.out.println("Number of queries: " + query);
      if( Label == InstancePair.MUST_LINK ){
	// update data structures
	m_NeighborSets[smallestCluster].add(new Integer(X));
	m_SumOfClusterInstances[smallestCluster] = sumWithInstance(m_SumOfClusterInstances[smallestCluster], m_Instances.instance(X));
	m_ClusterAssignments[X] = smallestCluster;
	if (m_verbose)
	  System.out.println("Adding " + X + " to cluster: " + smallestCluster);
	m_AssignedSet.add(new Integer(X));

	if( query >= numQueries ){
	  if (m_verbose)
	    System.out.println("Ran out of queries");
	  System.out.println("Consolidate phase over after " + query + " queries");	  
	  createGlobalCentroids();
	  addMLAndCLTransitiveClosure(null);
	  return;
	}
      }
      else { // must-link not found with smallest neighborhood, process other neighborhoods now
	if (m_verbose) 
	  System.out.println("Processing other centroids now");
	// compute centroids of other clusters
	for (int i=0; i<m_NumClusters; i++) { 
	  if (i != smallestCluster) { // already made query for smallest cluster
	    if (m_isSparseInstance) {
	      clusterCentroids[i] = new SparseInstance(m_SumOfClusterInstances[i]);
	    }
	    else {
	      clusterCentroids[i] = new Instance(m_SumOfClusterInstances[i]);
	    }
	    clusterCentroids[i].setDataset(m_Instances);
	    if (!m_objFunDecreasing) {
	      normalize(clusterCentroids[i]);
	    }
	    else {
	      normalizeByWeight(clusterCentroids[i]);
	    }
	  }
	}
	
	double[] similaritiesToCentroids = new double[m_NumClusters];
	for (int i=0; i<m_NumClusters; i++) {
	  if (i != smallestCluster) { // already made query for smallestCluster
	    similaritiesToCentroids[i] = m_metric.similarity(clusterCentroids[i], m_Instances.instance(X));
	  }
	} // handles both Euclidean and WeightedDotP
	
	if (m_verbose) {
	  System.out.println("Before sort");
	  for (int i=0; i<m_NumClusters; i++) {
	    System.out.println(similaritiesToCentroids[i]);
	  }
	}
	
	int[] indices = Utils.sort(similaritiesToCentroids); // sorts in ascending order of similarity
	
	if (m_verbose) {
	  System.out.println("After sort");
	  for (int i=0; i<m_NumClusters; i++) {
	    System.out.println(indices[i]);
	  }
	}
	
	for(int h = m_NumClusters-1; h >=0; h-- ){ 
	  // since sort is ascending, and we want descending sort of similarity values
	  int index = indices[h];
	  if (index != smallestCluster) { // already made query for smallest cluster
	    NbrIt = m_NeighborSets[index].iterator();
	    Y = ((Integer) NbrIt.next()).intValue(); // get any point from the neighborhood
	    Label = askOracle(X,Y);
	    query++;
	    System.out.println("Making query:" + query);
	    if (m_verbose) 
	      System.out.println("Number of queries: " + query);
	    if( Label == InstancePair.MUST_LINK ){
	      // update data structures
	      m_NeighborSets[index].add(new Integer(X));
	      m_SumOfClusterInstances[index] = sumWithInstance(m_SumOfClusterInstances[index], m_Instances.instance(X));
	      m_ClusterAssignments[X] = index;
	      if (m_verbose)
		System.out.println("Adding " + X + " to cluster: " + index);
	      m_AssignedSet.add(new Integer(X));
	      if (m_verbose)
		System.out.println("Exiting phase 2 for loop");
	      break; // exit from for
	    }
	    if( query >= numQueries ){
	      if (m_verbose)
		System.out.println("Ran out of queries");
	      createGlobalCentroids();
	      addMLAndCLTransitiveClosure(null);
	      return;
	    }
	  }
	} // end reverse for
      } // end else
    } // end while
    createGlobalCentroids();
    addMLAndCLTransitiveClosure(null);
    return;
  }

  /** Phase 2 code for active learning, random */

  protected void activePhaseTwoRandom(int numQueries) throws Exception {
    int numInstances = m_Instances.numInstances();
    int X,Y;
    int query = 0, Label;

    System.out.println("In Phase 2 with random, with numqueries: " + numQueries);
    while( query < numQueries ){
      if (m_verbose)
	System.out.println("Starting phase 2");
      
      Instance[] clusterCentroids = new Instance[m_NumClusters];
      if (m_AssignedSet.size() == m_StartingIndexOfTest) {
	if (m_verbose)
	  System.out.println("No more points left unassigned, we are DONE!!");
	createGlobalCentroids();
	addMLAndCLTransitiveClosure(null);
	return;
      }
      // find next point at random
      X = m_RandomNumberGenerator.nextInt(m_StartingIndexOfTest);
      while (m_AssignedSet != null && m_AssignedSet.contains(new Integer(X))) {
	X = m_RandomNumberGenerator.nextInt(m_StartingIndexOfTest);
      }
      if (m_verbose) 
	System.out.println("X = " + X + ", finding distances to centroids now");
      // compute centroids of other clusters
      for (int i=0; i<m_NumClusters; i++) { 
	if (m_isSparseInstance) {
	  clusterCentroids[i] = new SparseInstance(m_SumOfClusterInstances[i]);
	}
	else {
	  clusterCentroids[i] = new Instance(m_SumOfClusterInstances[i]);
	}
	clusterCentroids[i].setDataset(m_Instances);
	if (!m_objFunDecreasing) {
	  normalize(clusterCentroids[i]);
	}
	else {
	  normalizeByWeight(clusterCentroids[i]);
	}
      }
    
      double[] similaritiesToCentroids = new double[m_NumClusters];
      for (int i=0; i<m_NumClusters; i++) {
	similaritiesToCentroids[i] = m_metric.similarity(clusterCentroids[i], m_Instances.instance(X));
      } // handles both Euclidean and WeightedDotP
	
      if (m_verbose) {
	System.out.println("Before sort");
	for (int i=0; i<m_NumClusters; i++) {
	  System.out.println(similaritiesToCentroids[i]);
	}
      }
      
      int[] indices = Utils.sort(similaritiesToCentroids);
      
      if (m_verbose) {
	System.out.println("After sort");
	for (int i=0; i<m_NumClusters; i++) {
	  System.out.println(indices[i]);
	}
      }
      
      for(int h = m_NumClusters-1; h >=0; h-- ){ 
	// since sort is ascending, and we want descending sort of similarity values
	int index = indices[h];
	Iterator NbrIt = m_NeighborSets[index].iterator();
	Y = ((Integer) NbrIt.next()).intValue(); // get any point from neighborhood
	Label = askOracle(X,Y);
	query++;
	System.out.println("Making query:" + query);
	if (m_verbose) 
	  System.out.println("Number of queries: " + query);
	if( Label == InstancePair.MUST_LINK ){
	  // update data structures
	  m_NeighborSets[index].add(new Integer(X));
	  m_SumOfClusterInstances[index] = sumWithInstance(m_SumOfClusterInstances[index], m_Instances.instance(X));
	  m_ClusterAssignments[X] = index;
	  if (m_verbose)
	    System.out.println("Adding " + X + " to cluster: " + index);
	  m_AssignedSet.add(new Integer(X));
	  if (m_verbose)
	    System.out.println("Exiting phase 2 for loop");
	  break; // exit from for
	}
	else {
	  if (m_verbose)
	    System.out.println(X + " is CANNOT-LINKed to cluster " + index);
	}

	if( query >= numQueries ){
	  if (m_verbose)
	    System.out.println("Ran out of queries");
	  createGlobalCentroids();
	  addMLAndCLTransitiveClosure(null);
	  return;
	}
      }
    } // end reverse for
    createGlobalCentroids();
    addMLAndCLTransitiveClosure(null);
    return;
  }

  /** Creates the global cluster centroid */
  protected void createGlobalCentroids() throws Exception {
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
      if (m_isSparseInstance) {
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

      // normalize before random perturbation
      if (!m_objFunDecreasing) {
	normalizeInstance(m_GlobalCentroid);
      }
      
      System.out.println("Creating " + (m_NumClusters - m_NumCurrentClusters) + " random centroids");
      for (int i=m_NumCurrentClusters; i<m_NumClusters; i++) {
	double [] values = new double[m_Instances.numAttributes()];
	double normalizer = 0;
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  values[j] = m_GlobalCentroid.value(j) * (1 + m_DefaultPerturb * (m_RandomNumberGenerator.nextFloat() - 0.5));
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
    // add all ML links within clusters
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
	      InstancePair pair = null;
	      pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
	      if (!m_ConstraintsHash.containsKey(pair)) {
		m_ConstraintsHash.put(pair, new Integer(InstancePair.MUST_LINK));
		if (m_verbose) {
		  System.out.println("Adding inferred ML (" + pair.first +","+pair.second+")");
		}
		
		// hash the constraints for the instances involved
		Integer firstInt = new Integer(first);
		Integer secondInt = new Integer(second);
		InstancePair pairML = null;
		pairML = new InstancePair(first, second, InstancePair.MUST_LINK);
		Object constraintList1 = m_instanceConstraintHash.get(firstInt);
		if (constraintList1 == null) {
		  ArrayList constraintList = new ArrayList();
		  constraintList.add(pairML);
		  m_instanceConstraintHash.put(firstInt, constraintList);
		} else {
		  ((ArrayList)constraintList1).add(pairML);
		}
		Object constraintList2 = m_instanceConstraintHash.get(secondInt);
		if (constraintList2 == null) {
		  ArrayList constraintList = new ArrayList();
		  constraintList.add(pairML);
		  m_instanceConstraintHash.put(secondInt, constraintList);
		} else {
		  ((ArrayList)constraintList2).add(pairML);
		}
		
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
	for (int jj=ii+1; jj<m_NumCurrentClusters; jj++) {
	  int j = jj;
	  if (indices != null) {
	    j = indices[jj];
	  }
	  // check if there is at least one CL between neighborhoods ii & jj
	  boolean existsCL = false;

	  Iterator iter1 = m_NeighborSets[i].iterator();	
	  while (iter1.hasNext()) {
	    int index1 = ((Integer) iter1.next()).intValue();
	    if (m_NeighborSets[j] != null) {
	      Iterator iter2 = m_NeighborSets[j].iterator();
	      while (iter2.hasNext()) {
		int index2 = ((Integer) iter2.next()).intValue();
		int first = (index1 < index2)? index1:index2;
		int second = (index1 >= index2)? index1:index2;
		if (first == second) {
		  throw new Exception(" Same instance " + first + " cannot be in cluster: " + i + " and cluster " + j);
		}
		else if (first < second) {
		  InstancePair pair;
		  pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
		  if (m_ConstraintsHash.containsKey(pair)) {
		    // found one CL between the neighborhoods		    
		    existsCL = true;
		    break; // out of inner while
		  }
		}
	      }
	    }
	    if (existsCL) {
	      break; // out of outer while
	    }
	  }

	  // now add the inferred CLs
	  if (existsCL) {
	    iter1 = m_NeighborSets[i].iterator();	
	    while (iter1.hasNext()) {
	      int index1 = ((Integer) iter1.next()).intValue();
	      if (m_NeighborSets[j] != null) {
		Iterator iter2 = m_NeighborSets[j].iterator();
		while (iter2.hasNext()) {
		  int index2 = ((Integer) iter2.next()).intValue();
		  int first = (index1 < index2)? index1:index2;
		  int second = (index1 >= index2)? index1:index2;
		  if (first == second) {
		    throw new Exception(" Same instance " + first + " cannot be in cluster: " + i + " and cluster " + j);
		  }
		  else if (first < second) {
		    InstancePair pair;
		    pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);

		    // add new constraint
		    if (!m_ConstraintsHash.containsKey(pair)) {
		      m_ConstraintsHash.put(pair, new Integer(InstancePair.CANNOT_LINK));
		      if (m_verbose) {
			System.out.println("Adding inferred CL (" + pair.first +","+pair.second+")");
		      }

		      // hash the constraints for the instances involved
		      Integer firstInt = new Integer(first);
		      Integer secondInt = new Integer(second);
		      InstancePair pairCL;
		      pairCL = new InstancePair(first, second, InstancePair.CANNOT_LINK);
		      Object constraintList1 = m_instanceConstraintHash.get(firstInt);
		      if (constraintList1 == null) {
			ArrayList constraintList = new ArrayList();
			constraintList.add(pairCL);
			m_instanceConstraintHash.put(firstInt, constraintList);
		      } else {
			((ArrayList)constraintList1).add(pairCL);
		      }
		      Object constraintList2 = m_instanceConstraintHash.get(secondInt);
		      if (constraintList2 == null) {
			ArrayList constraintList = new ArrayList();
			constraintList.add(pairCL);
			m_instanceConstraintHash.put(secondInt, constraintList);
		      } else {
			((ArrayList)constraintList2).add(pairCL);
		      }
		    
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
	if(vertexColor[j] == WHITE){        // if the vertex is still undiscovered
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

    //    System.out.println("Need to make " + m_NumClusters + " clusters, already made " + m_NumCurrentClusters);

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
	System.out.println("Neighborhood selected:  " + m_NeighborSets[i].size() + "(" + m_TotalTrainWithLabels.instance(((Integer)(m_NeighborSets[i].iterator().next())).intValue()).classValue()+ ")\t");
	
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
      createGlobalCentroids();
      addMLAndCLTransitiveClosure(null);
    }
  }

  // Query: oracle replies on link, added to m_ConstraintsHash
  protected int askOracle(int X, int Y) {
    Instance first = m_TotalTrainWithLabels.instance(X);
    Instance second = m_TotalTrainWithLabels.instance(Y);
    int linkType;

    if (m_verbose) {
      System.out.print("["+X+","+Y);
    }
    if (first.classValue() == second.classValue()) {
      if (m_verbose) {
	System.out.println(",MUST]");
      }
      linkType = InstancePair.MUST_LINK;
    }
    else if (first.classValue() != second.classValue()) {
      if (m_verbose) {
	System.out.println(",CANNOT]");
      }
      linkType = InstancePair.CANNOT_LINK;
    } else {
      if (m_verbose) {
	System.out.println(",DONT_CARE]");
      }
      linkType = InstancePair.DONT_CARE_LINK;
    }
    
    // add to constraintHash and seedHash

    int firstIndex = (X<Y)? X:Y;
    int secondIndex = (X>=Y)? X:Y;
    InstancePair newPair = new InstancePair(firstIndex, secondIndex, InstancePair.DONT_CARE_LINK);
    if (!m_ConstraintsHash.containsKey(newPair)) {
      m_ConstraintsHash.put(newPair, new Integer(linkType));
    }

    Integer firstInt = new Integer(firstIndex);
    Integer secondInt = new Integer(secondIndex);
    // for first point 
    if(!m_SeedHash.contains(firstInt)) { // add instances with constraints to seedHash
      m_SeedHash.add(firstInt);
    }

    // for second point 
    if(!m_SeedHash.contains(secondInt)) {
      m_SeedHash.add(secondInt);
    }
    
    return linkType;
  }

  /** Finds point which has max min-distance from set visitedPoints, does not consider
      points from eliminationSet
   */
  int farthestFromSet(HashSet visitedPoints, HashSet eliminationSet) throws Exception {
    
    // implements farthest-first search algorithm:
    /*
      for (each datapoint x not in visitedPoints) {
        distance of x to visitedPoints = min{d(x,f):f \in visitedPoints}
      }
      select the point x with maximum distance as new center;
    */

    if (visitedPoints.size() == 0) {
      int point;
      if (m_StartingIndexOfTest < m_Instances.numInstances()) {	
	point = m_RandomNumberGenerator.nextInt(m_StartingIndexOfTest); // takes care not to select test example
      }
      else {
	point = m_RandomNumberGenerator.nextInt(m_Instances.numInstances());
      }
      // Note: no need to check for labeled data now, since we have no visitedPoints
      // => no labeled data
      if (m_verbose)
	System.out.println("First point selected: " + point);
      return point;
    }
    else {
      if (m_verbose) {
	Iterator iter = visitedPoints.iterator();
	if (eliminationSet != null) { 
	  iter = eliminationSet.iterator();
	  while(iter.hasNext()) {
	    System.out.println("In elimination set: " + ((Integer) iter.next()).intValue());
	  }
	}
      }
    }
    
    double minSimilaritySoFar = Double.POSITIVE_INFINITY;
    double maxDistanceSoFar = Double.NEGATIVE_INFINITY;
    ArrayList bestPointArray = null;
    int bestPoint = -1;

    for (int i=0; i<m_Instances.numInstances() && i<m_StartingIndexOfTest; i++) { // point should not belong to test set
      if (visitedPoints == null || !visitedPoints.contains(new Integer(i))) { // point should not belong to visitedPoints
	if (eliminationSet == null || !eliminationSet.contains(new Integer(i))) { // point should not belong to eliminationSet
	  Instance inst = m_Instances.instance(i);
	  Iterator iter = visitedPoints.iterator();
	  double minDistanceFromSet = Double.POSITIVE_INFINITY;
	  double maxSimilarityFromSet = Double.NEGATIVE_INFINITY;
	  while (iter.hasNext()) {
	    Instance pointInSet = m_Instances.instance(((Integer) iter.next()).intValue());
	    if (!m_objFunDecreasing) {
	      double sim = m_metric.similarity(inst, pointInSet);
	      if (sim > maxSimilarityFromSet) {
		maxSimilarityFromSet = sim;
	      }
	    }
	    else {
	      double dist = m_metric.distance(inst, pointInSet);
	      if (dist < minDistanceFromSet) {
		minDistanceFromSet = dist;
	      }
	    }
	  }
	  if (!m_objFunDecreasing) {
	    if (maxSimilarityFromSet == minSimilaritySoFar) {
	      minSimilaritySoFar = maxSimilarityFromSet;
	      bestPointArray.add(new Integer(i));
	    }
	    else if (maxSimilarityFromSet < minSimilaritySoFar) {
	      minSimilaritySoFar = maxSimilarityFromSet;
	      bestPointArray = new ArrayList();
	      bestPointArray.add(new Integer(i));
	    }
	  }
	  else {
	    if (minDistanceFromSet == maxDistanceSoFar) {
	      minDistanceFromSet = maxDistanceSoFar;
	      bestPointArray.add(new Integer(i));
	      if (m_verbose) {
		System.out.println("Additional point added: " + i + " with similarity: " + minSimilaritySoFar);
	      }
	    }
	    else if (minDistanceFromSet > maxDistanceSoFar) {
	      maxDistanceSoFar = minDistanceFromSet;
	      bestPointArray = new ArrayList();
	      bestPointArray.add(new Integer(i));
	      if (m_verbose) {
		System.out.println("Farthest point from set is: " + i + " with distance: " + maxDistanceSoFar);
	      }
	    }
	  }
	}
      }
    }

    if (bestPointArray == null) {
      System.out.println("\n\nAttention!! No more points left, all assigned\n\n");
    } else {
      if (m_verbose)
	System.out.println("Have " + bestPointArray.size() + " points in bestPointArray");
      int index = m_RandomNumberGenerator.nextInt(bestPointArray.size()); // select one of the bestPoints at random
      bestPoint = ((Integer) bestPointArray.get(index)).intValue();
    }

    if (m_verbose) {
      if (!m_objFunDecreasing) {
	System.out.println("Selected " + bestPoint + " with similarity: " + minSimilaritySoFar);
      }
      else {
	System.out.println("Selected " + bestPoint + " with distance: " + maxDistanceSoFar);
      }
    }
    return bestPoint;
  }


  /** Finds point which is nearest to center. This point should not be
   *  a test point and should not belong to visitedPoints 
   */
  int nearestFromPoint(Instance center, HashSet visitedPoints) throws Exception {
    double maxSimilarity = Double.NEGATIVE_INFINITY;
    double minDistance = Double.POSITIVE_INFINITY;
    int bestPoint = -1;

    for (int i=0; i<m_Instances.numInstances() && i<m_StartingIndexOfTest; i++) { // bestPoint should not be a test point
      if (!visitedPoints.contains(new Integer(i))) { // bestPoint should not belong to visitedPoints
	Instance inst = m_Instances.instance(i);
	if (!m_objFunDecreasing) {
	  double sim = m_metric.similarity(inst, center);
	  if (sim > maxSimilarity) {
	    bestPoint = i;
	    maxSimilarity = sim;
	  }
	}
	else {
	  double dist = m_metric.distance(inst, center);
	  if (dist < minDistance) {
	    bestPoint = i;
	    minDistance = dist;
	    if (m_verbose) {
	      System.out.println("Nearest point is: " + bestPoint + " with dist: " + minDistance);
	    }
	  }
	}
      }
    }
    return bestPoint;
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


  /** Updates the clusterAssignments for all points after clustering.
   *  Map assignments from [0,numInstances-1] to [0,numClusters-1]
   *  i.e. from [0 2 2 0 6 6 2] -> [0 1 1 0 2 2 0] 
   *  **** NOTE: THIS FUNCTION IS NO LONGER USED!!! ****
   */
  protected void updateClusterAssignments() throws Exception {

    //  **** DEPRECATED: THIS FUNCTION IS NO LONGER USED!!! ****

    int numInstances = m_Instances.numInstances();
    HashMap clusterNumberHash = new HashMap((int) (m_NumClusters/0.75+10));
    int clusterNumber = 0;

    if (m_verbose) {
      System.out.println("Mapping cluster assignments. Initial cluster assignments:");
      for (int i=0; i<numInstances; i++) {
	System.out.print(m_ClusterAssignments[i] + " ");
      }
      System.out.println();
    }
    
    for (int i=0; i<numInstances; i++) {
      if (m_ClusterAssignments[i]!=-1) {
	Integer clusterNum = new Integer(m_ClusterAssignments[i]);
	if (!clusterNumberHash.containsKey(clusterNum)) {
	  clusterNumberHash.put(clusterNum, new Integer(clusterNumber));
	  clusterNumber++;
	}
      }
    }
    if (clusterNumber != m_NumClusters) {
      throw new Exception("Number of clusters do not match");
    }
    for (int i=0; i<numInstances; i++) {
      if (m_ClusterAssignments[i]!=-1) {
	int newCluster = ((Integer) clusterNumberHash.get(new Integer(m_ClusterAssignments[i]))).intValue();
	m_ClusterAssignments[i] = newCluster;
      }
    }
    if (m_verbose) {
      System.out.println("Done updating cluster assignments. New cluster assignments:");
      for (int i=0; i<numInstances; i++) {
	System.out.print(m_ClusterAssignments[i] + " ");
      }
      System.out.println();
    }
    clusterNumberHash.clear(); clusterNumberHash = null; //free memory
  }

  /** Outputs the current clustering
   *
   * @exception Exception if something goes wrong
   */
  public void printIndexClusters() throws Exception {
    if (m_IndexClusters == null)
      throw new Exception ("Clusters were not created");

    for (int i = 0; i < m_NumClusters; i++) {
      HashSet cluster = m_IndexClusters[i];
      if (cluster == null) {
	System.out.println("Cluster " + i + " is null");
      }
      else {
	System.out.println ("Cluster " + i + " consists of " + cluster.size() + " elements");
	Iterator iter = cluster.iterator();
	while(iter.hasNext()) {
	  int idx = ((Integer) iter.next()).intValue();
	  System.out.println("\t\t" + idx);
	}
      }
    }
  }


  /** E-step of the KMeans clustering algorithm -- find best cluster assignments
   */
  protected void findBestAssignments() throws Exception{
    int moved = 0;

    int numInstances = m_Instances.numInstances();
    int [] indices = new int[numInstances];
    for (int i=0; i<numInstances; i++) {
      indices[i] = i; // initialize
    }
    
    if (m_InstanceOrdering == ORDERING_DEFAULT) {
      for (int i=0; i<numInstances; i++) {
	try {
	  // Update number of points moved
	  moved += assignInstanceToClusterWithConstraints(i);
	}
	catch (Exception e) {
	  System.out.println("Could not find distance. Exception: " + e);
	  e.printStackTrace();
	}
      }

      if (m_MovePointsTillAssignmentStabilizes) {
	int newMoved = -1;
	for (int t=0; t<100 && newMoved != 0; t++) { // move points till assignment stabilizes
	  newMoved = 0;
	  for (int i=0; i<numInstances; i++) {
	    newMoved += assignInstanceToClusterWithConstraints(i);
	  }
	  if (newMoved > 0) {
	    System.out.println(newMoved + " points moved on changing order in t=" + t);
	  } 
	}
      }
    } else if (m_InstanceOrdering == ORDERING_RANDOM) { // randomize instance ordering

      m_RandomNumberGenerator = new Random(m_RandomSeed); // initialize random number generator again
      for (int i = numInstances - 1; i > 0; i--) { 
	int indexToSwap = m_RandomNumberGenerator.nextInt(i+1);
	int temp = indices[i]; // swap
	indices[i] = indices[indexToSwap];
	indices[indexToSwap] = temp;
      }

      for (int i=0; i<numInstances; i++) {
	try {
	  // Update number of points moved
	  moved += assignInstanceToClusterWithConstraints(indices[i]);
	}
	catch (Exception e) {
	  System.out.println("Could not find distance. Exception: " + e);
	  e.printStackTrace();
	}
      }

      if (m_MovePointsTillAssignmentStabilizes) {
	int newMoved = -1;
	for (int t=0; t<100 && newMoved != 0; t++) { // move points till assignment stabilizes
	  newMoved = 0;
	  for (int i=0; i<numInstances; i++) {
	    newMoved += assignInstanceToClusterWithConstraints(indices[i]);
	  }
	  if (newMoved > 0) {
	    System.out.println(newMoved + " points moved on changing order in t=" + t);
	  } 
	}
      }
    } else if (m_InstanceOrdering == ORDERING_SORTED) {
      
      int [] sortedIndices = null;
      double bestSquareDistance = Integer.MAX_VALUE;
      double bestSimilarity = Integer.MIN_VALUE;

      double [] distances = new double[numInstances];

      // find closest cluster centroid for each instance
      for (int i = 0; i < numInstances; i++) {
	for (int j = 0; j < m_NumClusters; j++) {
	  double squareDistance = 0, similarity = 0;
	  if (!m_objFunDecreasing) {
	    similarity = similarityInPottsModel(i,j);
	    if (similarity > bestSimilarity) {
	      bestSimilarity = similarity;
	      distances[i] = -similarity; // hacked distance conversion for sorting
	    }
	  } else {
	    squareDistance = squareDistanceInPottsModel(i,j);
	    if (squareDistance < bestSquareDistance) {
	      bestSquareDistance = squareDistance;
	      distances[i] = squareDistance;
	    }
	  }
	}
      }

      sortedIndices = Utils.sort(distances); // sort in ascending order
      
      for (int i=0; i<numInstances; i++) {
	try {
	  // Update number of points moved
	  moved += assignInstanceToClusterWithConstraints(sortedIndices[i]);
	}
	catch (Exception e) {
	  System.out.println("Could not find distance. Exception: " + e);
	  e.printStackTrace();
	}
      }

      if (m_MovePointsTillAssignmentStabilizes) {
	int newMoved = -1;
	for (int t=0; t<100 && newMoved != 0; t++) { // move points till assignment stabilizes
	  newMoved = 0;
	  for (int i=0; i<numInstances; i++) {
	    newMoved += assignInstanceToClusterWithConstraints(sortedIndices[i]);
	  }
	  if (newMoved > 0) {
	    System.out.println(newMoved + " points moved on changing order in t=" + t);
	  } 
	}
      }
    } else {
      throw new Exception ("Unknown instance ordering!!");
    }
    
    System.out.println("\t" + moved + " points moved in this E-step");
  }

  /**
   * Classifies the instance using the current clustering considering
   * constraints, updates cluster assignments
   *
   * @param instance the instance to be assigned to a cluster
   * @return 1 if the point is moved, 0 otherwise
   * @exception Exception if instance could not be classified
   * successfully */

  public int assignInstanceToClusterWithConstraints(int instIdx) throws Exception {
    int bestCluster = 0;
    double bestSquareDistance = Integer.MAX_VALUE;
    double bestSimilarity = Integer.MIN_VALUE;
    int moved = 0;
    
    for (int i = 0; i < m_NumClusters; i++) {
      double squareDistance = 0, similarity = 0;
      if (!m_objFunDecreasing) {
	similarity = similarityInPottsModel(instIdx, i);
	//	System.out.println("Sim between instance " + instIdx + " and cluster " + i + " = " + similarity);
	if (similarity > bestSimilarity) {
	  bestSimilarity = similarity;
	  bestCluster = i;
	}
      } else {
	squareDistance = squareDistanceInPottsModel(instIdx, i);
	if (squareDistance < bestSquareDistance) {
	  bestSquareDistance = squareDistance;
	  bestCluster = i;
	}
      }
    }
    if (m_ClusterAssignments[instIdx] != bestCluster) {
      if (m_verbose) {
	System.out.println("Moving instance " + instIdx + " from cluster " + m_ClusterAssignments[instIdx] + " to cluster " + bestCluster);
      }
      moved = 1;

//        // remove instIdx from old cluster
//        if (m_ClusterAssignments[instIdx] < m_NumClusters && m_ClusterAssignments[instIdx] != -1 && m_IndexClusters[m_ClusterAssignments[instIdx]] != null ) {
//  	m_IndexClusters[m_ClusterAssignments[instIdx]].remove(new Integer(instIdx));
//        }
//        // add instIdx to new cluster
//        if (m_IndexClusters[bestCluster] == null) {
//  	m_IndexClusters[bestCluster] = new HashSet();
//        }
//        m_IndexClusters[bestCluster].add(new Integer (instIdx));

      // updates cluster Assignments
      m_ClusterAssignments[instIdx] = bestCluster; 
    }
    if (m_verbose) {
      System.out.println("Assigning instance " + instIdx + " to cluster " + bestCluster);
    }

    return moved;
  }

  /** finds similarity between instance and centroid in Potts Model
   */
  double similarityInPottsModel(int instIdx, int centroidIdx) throws Exception{
    double sim = m_metric.similarity(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));

    Object list =  m_instanceConstraintHash.get(new Integer(instIdx));
    if (list != null) {   // there are constraints associated with this instance
      ArrayList constraintList = (ArrayList) list;
      for (int i = 0; i < constraintList.size(); i++) {
	InstancePair pair = (InstancePair) constraintList.get(i);
	int firstIdx = pair.first;
	int secondIdx = pair.second;
	Instance instance1 = m_Instances.instance(firstIdx);
	Instance instance2 = m_Instances.instance(secondIdx);
	int otherIdx = (firstIdx == instIdx) ? m_ClusterAssignments[secondIdx] : m_ClusterAssignments[firstIdx];
	
	// check whether the constraint is violated
	if (otherIdx != -1) { 
	  if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) { 
	    sim -= m_MustLinkWeight;
	  } else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) { 
	    sim -= m_CannotLinkWeight;
	  }
	}
      }
    }

    if(m_verbose) {
      System.out.println("Final similarity between instance " + instIdx + " and centroid " + centroidIdx + " is: " + sim);
    }
    return sim;
  }

  /** finds squaredistance between instance and centroid in Potts Model
   */
  double squareDistanceInPottsModel(int instIdx, int centroidIdx) throws Exception{
    double dist = m_metric.distance(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));
    dist *= dist; // doing the squaring here itself
    if(m_verbose) {
      System.out.println("Unconstrained distance between instance " + instIdx + " and centroid " + centroidIdx + " is: " + dist);
    }

    Object list =  m_instanceConstraintHash.get(new Integer(instIdx));
    if (list != null) {   // there are constraints associated with this instance
      ArrayList constraintList = (ArrayList) list;
      for (int i = 0; i < constraintList.size(); i++) {
	InstancePair pair = (InstancePair) constraintList.get(i);
	int firstIdx = pair.first;
	int secondIdx = pair.second;
	Instance instance1 = m_Instances.instance(firstIdx);
	Instance instance2 = m_Instances.instance(secondIdx);
	int otherIdx = (firstIdx == instIdx) ? m_ClusterAssignments[secondIdx] : m_ClusterAssignments[firstIdx];
	
	// check whether the constraint is violated
	if (otherIdx != -1) { 
	  if (otherIdx != centroidIdx && pair.linkType == InstancePair.MUST_LINK) { 
	    dist += m_MustLinkWeight;
	  } else if (otherIdx == centroidIdx && pair.linkType == InstancePair.CANNOT_LINK) { 
	    dist += m_CannotLinkWeight;
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
    
    for (int i = 0; i < m_NumClusters; i++) {
      tempI[i] = new Instances(m_Instances, 0); // tempI[i] stores the cluster instances for cluster i
    }
    for (int i = 0; i < m_Instances.numInstances(); i++) {
      tempI[m_ClusterAssignments[i]].add(m_Instances.instance(i));
    }
    
    // Calculates cluster centroids
    for (int i = 0; i < m_NumClusters; i++) {
      double [] values = new double[m_Instances.numAttributes()];
      if (m_isSparseInstance) {
	values = meanOrMode(tempI[i]); // uses fast meanOrMode
      }
      else {
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  values[j] = tempI[i].meanOrMode(j); // uses usual meanOrMode
	}
      }
      
      // cluster centroids are dense in SPKMeans
      m_ClusterCentroids.add(new Instance(1.0, values));
      if (m_Algorithm == ALGORITHM_SPHERICAL) {
	try {
	  normalize(m_ClusterCentroids.instance(i));
	}
	catch (Exception e) {
	  e.printStackTrace();
	}
      }
    }
    for (int i = 0; i < m_NumClusters; i++) {
      tempI[i] = null; // free memory for garbage collector to pick up
    }
  }
  
  /** calculates objective function */
  protected void calculateObjectiveFunction() throws Exception {
    if (m_verbose) {
      System.out.println("Calculating objective function ...");
    }
    m_Objective = 0;

    double tempML = m_MustLinkWeight;
    double tempCL = m_CannotLinkWeight;
    m_MustLinkWeight = tempML/2;
    m_CannotLinkWeight = tempCL/2; // adjust weights to take care of double counting of constraints 
    if (m_verbose) {
      System.out.println("Must link weight: " + m_MustLinkWeight);
      System.out.println("Cannot link weight: " + m_CannotLinkWeight);    
    }

    for (int i=0; i<m_Instances.numInstances(); i++) {
      if (m_objFunDecreasing) {
	m_Objective += squareDistanceInPottsModel(i, m_ClusterAssignments[i]);
      }
      else {
	m_Objective += similarityInPottsModel(i, m_ClusterAssignments[i]);
      }
    }
    double o = m_Objective;
    
    m_MustLinkWeight = tempML;
    m_CannotLinkWeight = tempCL; // reset the values of the constraint weights
    if (m_verbose) {
      System.out.println("Must link weight: " + m_MustLinkWeight);
      System.out.println("Cannot link weight: " + m_CannotLinkWeight);    
    }
  }

  
  /** Actual KMeans function */
  protected void runKMeans() throws Exception {
    boolean converged = false;
    m_Iterations = 0;

    double oldObjective = m_objFunDecreasing ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;

    while (!converged) {
      // E-step: updates m_Objective
      if (m_verbose) {
	System.out.println("Doing E-step ...");
      }
      // to find the instance indices in the clusters, for constraint calculation in E-step

      findBestAssignments();

      // Find objective function
      if (m_Iterations > 0) {
	calculateObjectiveFunction();
	System.out.println("Objective function after point point assignment: " + m_Objective);
      }

      // M-step
      if (m_verbose) {
	System.out.println("Doing M-step ...");
      }
      updateClusterCentroids(); 
      calculateObjectiveFunction();
      System.out.println("Objective function after centroid estimation: " + m_Objective);

      m_Iterations++;
      
      // Convergence check
      if(Math.abs(oldObjective - m_Objective) > m_ObjFunConvergenceDifference) {
	converged = false;
      }
      else {
	converged = true;
	System.out.println("Final Objective function is: " + m_Objective);
      }
      if ((!m_objFunDecreasing && oldObjective > m_Objective) ||
	  (m_objFunDecreasing && oldObjective < m_Objective)) {
	throw new Exception("Oscillations => bug in objective function/EM step!!");
      }
      oldObjective = m_Objective;
    }
  }

  /** Dummy: not implemented for PCKMeans */
  public int[] bestInstancesForActiveLearning(int numActive) throws Exception{
    throw new Exception("Not implemented for PCKMeans");
  }

  /** Returns the indices of the best numActive instances for active learning */
  public InstancePair[] bestPairsForActiveLearning(int numActive) throws Exception{
    int usedQueries = activePhaseOne(numActive);
    if (m_PhaseTwoRandom) {
      activePhaseTwoRandom(numActive-usedQueries);
    }
    else {
      activePhaseTwoRoundRobin(numActive-usedQueries);
    }
    return null;
  }


  /**
   * Checks if instance has to be normalized and classifies the
   * instance using the current clustering
   *
   * @param instance the instance to be assigned to a cluster
   * @return the number of the assigned cluster as an integer
   * if the class is enumerated, otherwise the predicted value
   * @exception Exception if instance could not be classified
   * successfully */

  public int clusterInstance(Instance instance) throws Exception {
    if (m_Algorithm == ALGORITHM_SPHERICAL) { // check here, since evaluateModel calls this function on test data
      normalize(instance);
    }
    return assignInstanceToCluster(instance);
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
    System.out.println("Something's wrong, were supposed to look up but couldn't find it; size=" +  m_checksumHash.size());
    throw new Exception("WARNING!!!\n\nCouldn't lookup the instance!!!!\n\n");
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


  /** Return m_PhaseTwoRandom */
  public  boolean getPhaseTwoRandom() {
    return m_PhaseTwoRandom;
  }

  /** Set m_PhaseTwoRandom */
  public void setPhaseTwoRandom(boolean w) {
    m_PhaseTwoRandom = w;
  }

  /** Return m_AllExplore */
  public  boolean getAllExplore() {
    return m_AllExplore;
  }

  /** Set m_AllExplore */
  public void setAllExplore(boolean b) {
    m_AllExplore = b;
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
   * Set m_MovePointsTillAssignmentStabilizes
   * @param b truth value
   */
  public void setMovePointsTillAssignmentStabilizes (boolean b) {
    m_MovePointsTillAssignmentStabilizes = b;
  }
    
  /** Return m_MovePointsTillAssignmentStabilizes */
  public boolean getMovePointsTillAssignmentStabilizes () {
    return  m_MovePointsTillAssignmentStabilizes;
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
    for (int i = 0; i < m_checksumCoeffs.length; i++) {
      m_checksumCoeffs[i] = m_RandomNumberGenerator.nextDouble();
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


  /**
   * Set the instance ordering
   *
   * @param order instance ordering
   */
  public void setInstanceOrdering (SelectedTag order)
  {
    if (order.getTags() == TAGS_ORDERING) {
      if (m_verbose) {
	System.out.println("Ordering: " + order.getSelectedTag().getReadable());
      }
      m_InstanceOrdering = order.getSelectedTag().getID();
    }
  }

  /**
   * Get the instance ordering
   *
   * @returns ordering type
   */
  public SelectedTag getInstanceOrdering ()
  {
    return new SelectedTag(m_InstanceOrdering, TAGS_ORDERING);
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

  public HashSet[] getIndexClusters() throws Exception {
    m_IndexClusters = new HashSet[m_NumClusters];
    for (int i=0; i < m_Instances.numInstances(); i++) {
      //        if (m_verbose) {
      //  	System.out.println("In getIndexClusters, " + i + " assigned to cluster " + m_ClusterAssignments[i]);
      //        }
      if (m_ClusterAssignments[i]!=-1 && m_ClusterAssignments[i] < m_NumCurrentClusters) {
	if (m_IndexClusters[m_ClusterAssignments[i]] == null) {
	  m_IndexClusters[m_ClusterAssignments[i]] = new HashSet();
	}
	m_IndexClusters[m_ClusterAssignments[i]].add(new Integer(i));
      }
    }
    return m_IndexClusters;
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
    
    if (m_MovePointsTillAssignmentStabilizes) {
      options[current++] = "-Stable";
    }
    
    options[current++] = "-IO";
    options[current++] = "" + getInstanceOrdering().getSelectedTag().getID();
      
    options[current++] = "-A";
    options[current++] = "" + getAlgorithm().getSelectedTag().getID();
    if (getActive()) {
      options[current++] = "-active";
    }
    options[current++] = "-N";
    options[current++] = "" + getNumClusters();
    options[current++] = "-E";
    options[current++] = "" + getAllExplore();
    options[current++] = "-P2";
    options[current++] = "" + getPhaseTwoRandom();
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
      setMetric((LearnableMetric)LearnableMetric.forName(metricName, metricSpec));
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
   * set the active level of the clusterer
   * @param active
   */
  public void setActive (boolean active) {
    m_Active = active;
  }

  /**
   * get the active level of clusterer
   * @return active
   */
  public boolean getActive () {
    return m_Active;
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
      //System.out.println(ClusterEvaluation.evaluateClusterer(new PCKMeans(), args));
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
	int numPairs = 100, num=0;
	ArrayList labeledPairs = new ArrayList(numPairs);
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
	  if (first!=second && !labeledPairs.contains(pair)) {
	    labeledPairs.add(pair);
	    num++;
	  }
	}
	System.out.println("Finished initializing constraints");
	
	// create clusterer
	PCKMeans pckmeans = new PCKMeans();
	System.out.println("\nClustering the iris data using PCKmeans...\n");
	pckmeans.setAlgorithm(new SelectedTag(ALGORITHM_SIMPLE, TAGS_ALGORITHM));
	WeightedEuclidean euclidean = new WeightedEuclidean();
	euclidean.setExternal(false);
	pckmeans.setMetric(euclidean);
	pckmeans.setVerbose(false);
	pckmeans.setActive(false);
	pckmeans.setSeedable(true);
	pckmeans.setNumClusters(data.numClasses());

	// do clustering
	pckmeans.buildClusterer(labeledPairs, clusterData, data, data.numInstances());
	pckmeans.getIndexClusters();
	//	pckmeans.printIndexClusters();

	SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(pckmeans.m_TotalTrainWithLabels,
									 pckmeans.m_TotalTrainWithLabels.numClasses(),
									 pckmeans.m_TotalTrainWithLabels.numClasses());
	eval.evaluateModel(pckmeans, pckmeans.m_TotalTrainWithLabels, pckmeans.m_Instances);
	System.out.println("MI=" + eval.mutualInformation());
	System.out.print("FM=" + eval.pairwiseFMeasure());
	System.out.print("\tP=" + eval.pairwisePrecision());
	System.out.print("\tR=" + eval.pairwiseRecall());
      }
      else if (dataset.equals("highd")) {
	//////// Newsgroup data
	//	String datafile = "/u/ml/data/CCSfiles/arffFromCCS/cmu-newsgroup-clean-1000_fromCCS.arff";
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
	ArrayList labeledPairs = new ArrayList(numPairs);
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
	  if (first!=second && !labeledPairs.contains(pair)) {
	    labeledPairs.add(pair);
	    num++;
	  }
	}
	System.out.println("Finished initializing constraints");
	
	// create clusterer
	PCKMeans pckmeans = new PCKMeans();
	System.out.println("\nClustering the news data using PCKmeans...\n");
	pckmeans.resetClusterer();
	pckmeans.setAlgorithm(new SelectedTag(ALGORITHM_SPHERICAL, TAGS_ALGORITHM));
	pckmeans.setInstanceOrdering(new SelectedTag(ORDERING_SORTED, TAGS_ORDERING));
	pckmeans.setMovePointsTillAssignmentStabilizes(false);
	WeightedDotP dotp = new WeightedDotP();
	dotp.setExternal(false);
	dotp.setLengthNormalized(true);
	pckmeans.setMetric(dotp);
	pckmeans.setVerbose(false);
	pckmeans.setActive(false);
	//pckmeans.setActive(true); // uncomment to run Active Learning
	pckmeans.setSeedable(true);
	pckmeans.setNumClusters(data.numClasses());

	// do clustering
	pckmeans.buildClusterer(labeledPairs, clusterData, data, clusterData.numInstances());
	pckmeans.getIndexClusters();
	//	pckmeans.printIndexClusters();

	SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(pckmeans.m_TotalTrainWithLabels,
									 pckmeans.m_TotalTrainWithLabels.numClasses(),
									 pckmeans.m_TotalTrainWithLabels.numClasses());
	eval.evaluateModel(pckmeans, pckmeans.m_TotalTrainWithLabels, pckmeans.m_Instances);
	System.out.println("MI=" + eval.mutualInformation());
	System.out.print("FM=" + eval.pairwiseFMeasure());
	System.out.print("\tP=" + eval.pairwisePrecision());
	System.out.print("\tR=" + eval.pairwiseRecall());
      }
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }
}

// TODO: Add reading constraints from file
// TODO: Add all the options to setOptions
// TODO: Add all the options in comment on top of class
