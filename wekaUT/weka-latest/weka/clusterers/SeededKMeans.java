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
 *    SeededKMeans.java
 *    Copyright (C) 2002 Sugato Basu 
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
 * Seeded k means clustering class.
 *
 * Valid options are:<p>
 *
 * -N <number of clusters> <br>
 * Specify the number of clusters to generate. <p>
 *
 * -R <random seed> <br>
 * Specify random number seed <p>
 *
 * -S <seeding method> <br>
 * The seeding method can be "seeded" (seeded KMeans) or "constrained" (constrained KMeans)
 *
 * -A <algorithm> <br>
 * The algorithm can be "simple" (simple KMeans) or "spherical" (spherical KMeans)
 *
 * -M <metric-class> <br>
 * Specifies the name of the distance metric class that should be used
 * 
 * @author Sugato Basu(sugato@cs.utexas.edu)
 * @see Clusterer
 * @see OptionHandler
 */
public class SeededKMeans extends Clusterer implements OptionHandler,SemiSupClusterer,ActiveLearningClusterer {

  /** Name of clusterer */
  String m_name = "SeededKMeans";

  /** holds the clusters */
  protected ArrayList m_FinalClusters = null;
  
  /** holds the instance indices in the clusters */
  protected ArrayList m_IndexClusters = null;

  /** holds the ([seed instance] -> [clusterLabel of seed instance]) mapping */
  protected HashMap m_SeedHash = null;

  /** distance Metric */
  protected Metric m_metric = new WeightedDotP();

  /** has the metric has been constructed?  a fix for multiple buildClusterer's */
  protected boolean m_metricBuilt = false;

  /** starting index of test data in unlabeledData if transductive clustering */
  protected int m_StartingIndexOfTest = -1;

  /** indicates whether instances are sparse */
  protected  boolean isSparseInstance = false;

  /** Is the objective function increasing or decreasing?  Depends on type
   * of metric used:  for similarity-based metric, increasing, for distance-based - decreasing */
  protected boolean m_objFunDecreasing = false;

  /** Name of metric */
  protected String m_metricName = new String("WeightedDotP");

  /** Points that are to be skipped in the clustering process
   * because they are collapsed to zero */
  protected HashSet m_skipHash = new HashSet();

  /** Index of the current element in the E-step */
  protected int m_currIdx = 0;

  /** keep track of the number of iterations completed before convergence
   */
  protected int m_Iterations = 0;

  /* Define possible seeding methods */
  public static final int SEEDING_CONSTRAINED = 1;
  public static final int SEEDING_SEEDED = 2;
  public static final Tag [] TAGS_SEEDING = {
    new Tag(SEEDING_CONSTRAINED, "Constrained seeding"),
    new Tag(SEEDING_SEEDED, "Initial seeding only")
      };
  
  /** seeding method, by default seeded */
  protected int m_SeedingMethod = SEEDING_SEEDED;
  
  /** Define possible algorithms */
  public static final int ALGORITHM_SIMPLE = 1;
  public static final int ALGORITHM_SPHERICAL = 2;
  public static final Tag[] TAGS_ALGORITHM = {
    new Tag(ALGORITHM_SIMPLE, "Simple K-Means"),
    new Tag(ALGORITHM_SPHERICAL, "Spherical K-Means")
      };
  
  /** algorithm, by default spherical */
  protected int m_Algorithm = ALGORITHM_SPHERICAL;
  
  /** min difference of objective function values for convergence*/
  protected double m_ObjFunConvergenceDifference = 1e-5;

  /** value of objective function */
  protected double m_Objective  = Integer.MAX_VALUE;

  /** returns objective function */
  public double objectiveFunction() {
    return m_Objective;
  }

  /** Verbose? */
  protected boolean m_Verbose = false;

  /**
   * training instances with labels
   */
  protected Instances m_TotalTrainWithLabels;

  /**
   * training instances
   */
  protected Instances m_Instances;

  /**
   * number of clusters to generate, default is 3
   */
  protected int m_NumClusters = 3;

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

  /** weight of the concentration */
  protected double m_Concentration = 10.0;

  /** number of extra phase1 runs */
  protected double m_ExtraPhase1RunFraction = 50;

  /**
   * temporary variable holding cluster assignments while iterating
   */
  protected int [] m_ClusterAssignments;


  /**
   * holds the random Seed, useful for randomPerturbInit
   */
  protected int m_randomSeed = 1;


  /** semisupervision */
  protected boolean m_Seedable = true;
  
  /* Constructor */
  public SeededKMeans() {
  }

  /* Constructor */
  public SeededKMeans(Metric metric) {
    m_metric = metric;
    m_metricName = m_metric.getClass().getName();
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
    setNumClusters(num_clusters);
    if (m_Algorithm == ALGORITHM_SPHERICAL && m_metric instanceof WeightedDotP) {
      ((WeightedDotP)m_metric).setLengthNormalized(false); // since instances and clusters are already normalized, we don't need to normalize again while computing similarity - saves time
    }
    if (data.instance(0) instanceof SparseInstance) {
      isSparseInstance = true;
    }    
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
   * @param startingIndexOfTest from where test data starts in unlabeledData, useful if clustering is transductive
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(Instances labeledData, Instances unlabeledData, int classIndex, int numClusters, int startingIndexOfTest) throws Exception {
    m_StartingIndexOfTest = startingIndexOfTest;
    buildClusterer(labeledData, unlabeledData, classIndex, numClusters);
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
  public void buildClusterer(Instances labeledData, Instances unlabeledData, int classIndex, Instances totalTrainWithLabels, int startingIndexOfTest) throws Exception {
    m_StartingIndexOfTest = startingIndexOfTest;
    m_TotalTrainWithLabels = totalTrainWithLabels;
    buildClusterer(labeledData, unlabeledData, classIndex, totalTrainWithLabels.numClasses());
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
  public void buildClusterer(Instances labeledData, Instances unlabeledData, int classIndex, int numClusters) throws Exception {
    if (m_Algorithm == ALGORITHM_SPHERICAL) {
      if (labeledData != null) {
	for (int i=0; i<labeledData.numInstances(); i++) {
	  normalize(labeledData.instance(i));
	}
      }
      for (int i=0; i<unlabeledData.numInstances(); i++) {
	normalize(unlabeledData.instance(i));
      }
    }

    Instances clusterData = new Instances(unlabeledData, 0);;
    
    if (getSeedable()) {
    // remove labels of labeledData before putting in seedHash
      clusterData = new Instances(labeledData);
      System.out.println("Numattributes: " + clusterData.numAttributes());
      clusterData.deleteClassAttribute();
      // create seedHash from labeledData
      Seeder seeder = new Seeder(clusterData, labeledData);
      setSeedHash(seeder.getAllSeeds());
    }

    // add unlabeled data to labeled data (labels removed), not the
    // other way around, so that the labels in the hash table entries
    // and m_TotalTrainWithLabels are consistent
    for (int i=0; i<unlabeledData.numInstances(); i++) {
      clusterData.add(unlabeledData.instance(i));
    }
    
    
    System.out.println("combinedData has size: " + clusterData.numInstances() + "\n");

    // learn metric using labeled data, then  cluster both the labeled and unlabeled data
    if (labeledData != null) {
      m_metric.buildMetric(labeledData);
    }
    else {
      m_metric.buildMetric(unlabeledData.numAttributes());
    }
    m_metricBuilt = true;
    buildClusterer(clusterData, numClusters);
  }

  /**
   * Reset all values that have been learned
   */
  public void resetClusterer()  throws Exception{
    if (m_metric instanceof LearnableMetric)
      ((LearnableMetric)m_metric).resetMetric();
    m_SeedHash = null;
    m_ClusterCentroids = null;
  }


  
  /**
   * We can have clusterers that don't utilize seeding
   */
  public boolean seedable() {
    return m_Seedable;
  }

  /** Initializes the cluster centroids - initial M step
   */
  protected void initializeClusterer() {
    Random random = new Random(m_randomSeed);
    boolean globalCentroidComputed = false;

    if (m_Verbose) {
      //      System.out.println("SeedHash is: " + m_SeedHash);
    }
        
    System.out.println("Initializing ");
    // makes initial cluster assignments
    for (int i = 0; i < m_Instances.numInstances(); i++) {
      Instance inst = m_Instances.instance(i);
      
      if (m_SeedHash != null && m_SeedHash.containsKey(inst)) {
	m_ClusterAssignments[i] = ((Integer) m_SeedHash.get(inst)).intValue();
	if (m_ClusterAssignments[i] < 0) {
	  m_ClusterAssignments[i] = -1; // For randomPerturbInit
	  if (m_Verbose) {
	    System.out.println("Invalid cluster specification for seed instance " + i + ": " + inst + ", making random initial assignment");
	  }
	}
	else {
	  if (m_Verbose) {
	    System.out.println("Seed instance " + i + ": " + inst + " assigned to cluster: " + m_ClusterAssignments[i]);
	  }
	}
      }
      else {
	m_ClusterAssignments[i] = -1; // For randomPerturbInit
      }
    }

    Instances [] tempI = new Instances[m_NumClusters];
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    boolean [] clusterSeeded = new boolean[m_NumClusters];
    
    for (int i = 0; i < m_NumClusters; i++) {
      tempI[i] = new Instances(m_Instances, 0); // tempI[i] stores the cluster instances for cluster i
      clusterSeeded[i] = false; // initialize all clusters to be unseeded
    }
    for (int i = 0; i < m_Instances.numInstances(); i++) {
      if (m_ClusterAssignments[i] >= 0) { // seeded cluster
	clusterSeeded[m_ClusterAssignments[i]] = true;
	tempI[m_ClusterAssignments[i]].add(m_Instances.instance(i));
      }
    }
    
    // Calculates initial cluster centroids
    for (int i = 0; i < m_NumClusters; i++) {
      double [] values = new double[m_Instances.numAttributes()];
      if (clusterSeeded[i] == true) {
 	if (m_FastMode && isSparseInstance) {
	  values = meanOrMode(tempI[i]); // uses fast meanOrMode
	}
	else {
	  for (int j = 0; j < m_Instances.numAttributes(); j++) {
	    values[j] = tempI[i].meanOrMode(j); // uses usual meanOrMode
	  }
	}
      }
      else {
	// finds global centroid if has not been already computed
	if (!globalCentroidComputed) {
	  double [] globalValues = new double[m_Instances.numAttributes()];
	  if (m_FastMode && isSparseInstance) {
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
	  if (m_Algorithm == ALGORITHM_SPHERICAL) {
	    try {
	      ((LearnableMetric)m_metric).normalizeInstanceWeighted(m_GlobalCentroid);	
	    }
	    catch (Exception e) {
	      e.printStackTrace();
	    }
	  }
	  globalCentroidComputed = true;
	  if (m_Verbose) {
	    System.out.println("Global centroid is: " + m_GlobalCentroid);
	  }
	}
	// randomPerturbInit
	if (m_Verbose) {
	  System.out.println("RandomPerturbInit seeding for centroid " + i);
	}
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  values[j] = m_GlobalCentroid.value(j) * (1 + m_DefaultPerturb * (random.nextFloat() - 0.5));
	}	
      }
      
      // cluster centroids are dense in SPKMeans
      m_ClusterCentroids.add(new Instance(1.0, values));
      if (m_Algorithm == ALGORITHM_SPHERICAL) {
	try {
	  ((LearnableMetric) m_metric).normalizeInstanceWeighted(m_ClusterCentroids.instance(i));
	}
	catch (Exception e) {
	  e.printStackTrace();
	}
      }
    }    
  }

  /** E-step of the KMeans clustering algorithm -- find best cluster assignments
   */
  protected void findBestAssignments() throws Exception{
    m_Objective = 0;
    int moved=0;

    for (int i = 0; i < m_Instances.numInstances(); i++) {
      m_currIdx = i;
      Instance inst = m_Instances.instance(i);
      boolean assigned = false;

      // Constrained KMeans algorithm
      if(m_SeedingMethod == SEEDING_CONSTRAINED) {
	if (m_SeedHash == null) {	    
	  System.err.println("Needs seed information for constrained SeededKMeans");
	}
	else if(m_SeedHash.containsKey(inst)) { // Seeded instances
	  m_ClusterAssignments[i] = ((Integer) m_SeedHash.get(inst)).intValue(); 
	  assigned = true;
	  if (m_Verbose) {
	    System.out.println("Assigning cluster " + m_ClusterAssignments[i] + " for seed instance " + i + ": " + inst);
	  }
	}
      }

      try {
	if (!assigned) { // Unseeded instances
	  int newAssignment = assignClusterToInstance(inst);
	  if (newAssignment != m_ClusterAssignments[i]) {
	    moved++;
	    if (m_Verbose) {
	      System.out.println("Reassigning instance " + i + " old cluster=" + m_ClusterAssignments[i] + " new cluster=" + newAssignment);
	    }
	  }
	  m_ClusterAssignments[i] = newAssignment;
	}

	// Update objective function
	if (!m_objFunDecreasing) { // objective function increases monotonically
	  double newSimilarity = m_metric.similarity(inst, m_ClusterCentroids.instance(m_ClusterAssignments[i]));
	  m_Objective += newSimilarity;
	} 
	else { // objective function decreases monotonically
	  double newDistance = m_metric.distance(inst, m_ClusterCentroids.instance(m_ClusterAssignments[i]));
	  m_Objective += newDistance * newDistance;
	}
      } 
      catch (Exception e) {
	System.out.println("Could not find distance. Exception: " + e);
	e.printStackTrace();
      }
    }
    
    if(m_Verbose) {
      System.out.println("\nAfter iteration " + m_Iterations + ":\n");
      /*
      for (int k=0; k<m_ClusterCentroids.numInstances(); k++) {
	System.out.println ("  Centroid " + k + " is " + m_ClusterCentroids.instance(k));
      }
      */
    }
    System.out.println("Number of points moved in this E-step: " + moved);
  }

  /** M-step of the KMeans clustering algorithm -- updates cluster centroids
   */
  protected void updateClusterCentroids() {
    // M-step: update cluster centroids
    Instances [] tempI = new Instances[m_NumClusters];
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    
    for (int i = 0; i < m_NumClusters; i++) {
      tempI[i] = new Instances(m_Instances, 0); // tempI[i] stores the cluster instances for cluster i
    }
    for (int i = 0; i < m_Instances.numInstances(); i++) {
      tempI[m_ClusterAssignments[i]].add(m_Instances.instance(i));
      if (m_Verbose) {
	System.out.println("Instance " + i + " added to cluster " + m_ClusterAssignments[i]);
      }
    }
    
    // Calculates cluster centroids
    for (int i = 0; i < m_NumClusters; i++) {
      double [] values = new double[m_Instances.numAttributes()];
      if (m_FastMode && isSparseInstance) {
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
	  ((LearnableMetric) m_metric).normalizeInstanceWeighted(m_ClusterCentroids.instance(i));
	}
	catch (Exception e) {
	  e.printStackTrace();
	}
      }
    }
  }

  /** calculates objective function */
  protected void calculateObjectiveFunction() throws Exception {
    m_Objective = 0;
    for (int i=0; i<m_Instances.numInstances(); i++) {
      if (m_objFunDecreasing) {
	double dist = m_metric.distance(m_Instances.instance(i), m_ClusterCentroids.instance(m_ClusterAssignments[i]));
	m_Objective += dist*dist;
      }
      else {
	//m_Objective += similarity(i, m_ClusterAssignments[i]);
	m_Objective += m_metric.similarity(m_Instances.instance(i), m_ClusterCentroids.instance(m_ClusterAssignments[i]));
      }
    }
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

    setInstances(data);
    // Don't rebuild the metric if it was already trained
    if (!m_metricBuilt) {
      m_metric.buildMetric(data);
    }
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    m_ClusterAssignments = new int [m_Instances.numInstances()];

    if (m_Verbose && m_SeedHash != null) {
      System.out.println("Using seeding ...");
    }

    if (m_Instances.checkForNominalAttributes() && m_Instances.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle nominal attributes\n");
    }

    initializeClusterer(); // Initializes cluster centroids (initial M-step)
    System.out.println("Done initializing clustering ...");
    getIndexClusters();
    printIndexClusters();
    if (m_Verbose) {
      for (int i=0; i<m_NumClusters; i++) {
	System.out.println("Centroid " + i + ": " + m_ClusterCentroids.instance(i));
      }
    }


    boolean converged = false;
    m_Iterations = 0;

    double oldObjective = m_objFunDecreasing ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;

    while (!converged) {
      // E-step: updates m_Objective
      System.out.println("Doing E-step ...");
      findBestAssignments();

      // M-step
      System.out.println("Doing M-step ...");
      updateClusterCentroids(); 
      m_Iterations++;

      calculateObjectiveFunction();
      // Convergence check
      if(Math.abs(oldObjective - m_Objective) > m_ObjFunConvergenceDifference) {
	if (m_objFunDecreasing ? (oldObjective <  m_Objective) : (oldObjective >  m_Objective)) {
	  converged = true;
	  System.out.println("\nOSCILLATING, oldObjective=" + oldObjective + " newObjective=" + m_Objective);
	  System.out.println("Seeding=" + m_Seedable + " SeedingMethod=" + m_SeedingMethod );	  
	} else {
	  converged = false;
	  System.out.println("Objective function is: " + m_Objective);
	}
      }
      else {
	converged = true;
	System.out.println("Old Objective function was: " + oldObjective);
	System.out.println("Final Objective function is: " + m_Objective);
      }
      oldObjective = m_Objective;
    }
  }

  public InstancePair[] bestPairsForActiveLearning(int numActive) throws Exception {
    throw new Exception("Not implemented for SeededKMeans");
  }

  /** Returns the indices of the best numActive instances for active learning */
  public int[] bestInstancesForActiveLearning(int numActive) throws Exception{
    int numInstances = m_Instances.numInstances();
    int [] clusterSizes = new int[m_NumClusters];
    int [] activeLearningPoints = new int[numActive];
    int [] clusterAssignments = new int[numInstances];
    Instance [] sumOfClusterInstances = new Instance[m_NumClusters];
    HashSet visitedPoints = new HashSet(numInstances);
    boolean allClustersFound = false;
    int numPointsSelected = 0;
    
    // initialize clusterAssignments, clusterSizes, visitedPoints, sumOfClusterInstances
    for (int i=0; i<numInstances; i++) {
      Instance inst = m_Instances.instance(i);
      if (m_SeedHash != null && m_SeedHash.containsKey(inst)) {
	clusterAssignments[i] = ((Integer) m_SeedHash.get(inst)).intValue(); 
	clusterSizes[clusterAssignments[i]]++;
	visitedPoints.add(new Integer(i));
	sumOfClusterInstances[clusterAssignments[i]] = sumWithInstance(sumOfClusterInstances[clusterAssignments[i]], inst);
	if (m_Verbose) {
	  //	  System.out.println("Init: adding point " + i + " to cluster " + clusterAssignments[i]);
	}
      }
      else {
	clusterAssignments[i] = -1;
      }
    }

    // set allClustersFound
    allClustersFound = setAllClustersFound(clusterSizes);
    int totalPointsSpecified=0;
    for (int i=0; i<m_NumClusters; i++) {
      totalPointsSpecified += clusterSizes[i]; // HACK!!!
    }
    System.out.println("Total points specified: " + totalPointsSpecified + ", limit: " + m_ExtraPhase1RunFraction);
    if (totalPointsSpecified < m_ExtraPhase1RunFraction) {
      allClustersFound = false;
    }
    
    while (numPointsSelected < numActive) {
      if (!allClustersFound) { // PHASE 1
	System.out.println("In Phase 1");
	// find next point, farthest from visited points
	int nextPoint = farthestFromSet(visitedPoints, null);
	if (nextPoint >= m_StartingIndexOfTest) {
	  throw new Exception ("Test point " + nextPoint + " selected, something went wrong -- starting index of test is: " + m_StartingIndexOfTest);
	}
	visitedPoints.add(new Integer(nextPoint));
	activeLearningPoints[numPointsSelected] = nextPoint;
	numPointsSelected++;
	// update cluster stats for this point
	int classLabel = (int) m_TotalTrainWithLabels.instance(nextPoint).classValue();
	clusterAssignments[nextPoint] = classLabel;
	clusterSizes[classLabel]++;
	sumOfClusterInstances[classLabel] = sumWithInstance(sumOfClusterInstances[classLabel], m_Instances.instance(nextPoint));
	// set allClustersFound
	//	if (m_Verbose) {
	System.out.println("Active learning point number: " + numPointsSelected + " is: " + nextPoint + ", with class label: " + classLabel);
	  //	}
	allClustersFound = setAllClustersFound(clusterSizes);
	if (numPointsSelected >= numActive) {
	  System.out.println("Out of queries before phase 1 extra loop. Queries so far: " + numPointsSelected);
	  return activeLearningPoints; // go out of function
	}

	if (allClustersFound) {
	  // Extra RUNS OF PHASE 1
	  int [] tempClusterSizes = new int[m_NumClusters]; // temp cluster sizes
	  boolean tempAllClustersFound = false;
	  HashSet points = new HashSet(numInstances); // points visited in this farthest first loop
	  points.add(new Integer(nextPoint)); // mark only last point as visited
	  tempClusterSizes[classLabel]++; // update temp cluster sizes for this point
	  HashSet eliminationSet = new HashSet(numInstances); // don't include these points in farthest first search
	  for (int i=0; i<numInstances; i++) {
	    Instance inst = m_Instances.instance(i);
	    if (m_SeedHash != null && m_SeedHash.containsKey(inst)) {
	      eliminationSet.add(new Integer(i)); // add labeled data to elimination set
	    }
	  }
	  Iterator iter = visitedPoints.iterator();
	  while(iter.hasNext()) {
	    eliminationSet.add(iter.next()); // add already visited points to elim set
	  }
	  for (int i=0; i<m_ExtraPhase1RunFraction; i++) {
	    System.out.println("Continuing Phase 1 run: " + i + " after all clusters visited");
	    // find next point, farthest from points, eliminating points in eliminationSet
	    nextPoint = farthestFromSet(points, eliminationSet);
	    if (nextPoint >= m_StartingIndexOfTest) {
	      throw new Exception ("Test point " + nextPoint + " selected, something went wrong -- starting index of test is: " + m_StartingIndexOfTest);
	    }
	    visitedPoints.add(new Integer(nextPoint)); // add to total set of visited points
	    points.add(new Integer(nextPoint)); // add to points visited in this farthest first loop
	    activeLearningPoints[numPointsSelected] = nextPoint;
	    numPointsSelected++;
	    // update cluster stats for this point
	    classLabel = (int) m_TotalTrainWithLabels.instance(nextPoint).classValue();
	    clusterAssignments[nextPoint] = classLabel;
	    clusterSizes[classLabel]++;
	    sumOfClusterInstances[classLabel] = sumWithInstance(sumOfClusterInstances[classLabel], m_Instances.instance(nextPoint));
	    tempClusterSizes[classLabel]++;
	    //	if (m_Verbose) {
	    System.out.println("Active learning point number: " + numPointsSelected + " is: " + nextPoint + ", with class label: " + classLabel);
	    //	}
	    tempAllClustersFound = setAllClustersFound(tempClusterSizes);
	    if (tempAllClustersFound) { // found all clusters, reset local variables
	      System.out.println("Resetting variables for next round of farthest first");
	      tempClusterSizes = new int[m_NumClusters];
	      tempAllClustersFound = false;
	      Iterator tempIter = points.iterator();
	      while(tempIter.hasNext()) {
		eliminationSet.add((Integer) tempIter.next()); // add already visited points to elim set
	      }
	      points.clear(); // clear current set
	      points.add(new Integer(nextPoint)); // add the last point
	      tempClusterSizes[classLabel]++; // for the last point
	    }
	    if (numPointsSelected >= numActive) {
	      System.out.println("Out of queries within phase 1 extra loop. Queries so far: " + numPointsSelected);
	      return activeLearningPoints; // go out of function
	    }
	  }
	}
      }
      else { // PHASE 2
	// find smallest cluster
	System.out.println("In Phase 2");
	int smallestSize = Integer.MAX_VALUE, smallestCluster = -1;
	for (int i=0; i<m_NumClusters; i++) {
	  if (clusterSizes[i] < smallestSize) {
	    smallestSize = clusterSizes[i];
	    smallestCluster = i;
	  }
	}
	if (m_Verbose) {
	  System.out.println("Smallest cluster now: " + smallestCluster + ", with size: " + smallestSize);
	}
	// compute centroid of smallest cluster
	Instance centroidOfSmallestCluster;
	if (isSparseInstance) {
	  centroidOfSmallestCluster = new SparseInstance(sumOfClusterInstances[smallestCluster]);
	}
	else {
	  centroidOfSmallestCluster = new Instance(sumOfClusterInstances[smallestCluster]);
	}
	centroidOfSmallestCluster.setDataset(m_Instances);
	if (!m_objFunDecreasing) {
	  normalize(centroidOfSmallestCluster);
	}
	else {
	  normalizeByWeight(centroidOfSmallestCluster);
	}
	// find next point, closest to centroid of smallest cluster
	int nextPoint = nearestFromPoint(centroidOfSmallestCluster, visitedPoints);
	if (nextPoint >= m_StartingIndexOfTest) {
	  throw new Exception ("Test point selected, something went wrong!");
	}
	visitedPoints.add(new Integer(nextPoint));
	activeLearningPoints[numPointsSelected] = nextPoint;
	numPointsSelected++;
	// update cluster stats for this point
	int classLabel = (int) m_TotalTrainWithLabels.instance(nextPoint).classValue();
	clusterAssignments[nextPoint] = classLabel;
	clusterSizes[classLabel]++;
	sumOfClusterInstances[classLabel] = sumWithInstance(sumOfClusterInstances[classLabel], m_Instances.instance(nextPoint));
	//	if (m_Verbose) {
	System.out.println("Active learning point number: " + numPointsSelected + " is: " + nextPoint + ", with class label: " + classLabel);
	  //	}
	allClustersFound = setAllClustersFound(clusterSizes);
	if (allClustersFound != true) {
	  throw new Exception("Something went wrong - all clusters should be set in phase 2!!");
	}
      }
    }
    return activeLearningPoints;
  }

  /** Returns true if all clusterSizes are non-zero 
   */
  boolean setAllClustersFound(int [] clusterSizes) {
    boolean found = true;
    for (int i=0; i<m_NumClusters; i++) {
      if (clusterSizes[i] == 0) {
	found = false;
      }
      //if (m_Verbose) {
      System.out.println("Cluster " + i + " has size: " + clusterSizes[i]);
      //}
    }
    return found;
  }

  /** Finds the sum of instance sum with instance inst 
   */
  Instance sumWithInstance(Instance sum, Instance inst) throws Exception {
    Instance newSum;
    if (sum == null) {
      if (isSparseInstance) {
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

  /** Finds point which has max min-distance from set visitedPoints 
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
      Random rand = new Random(m_randomSeed);
      int point = rand.nextInt(m_StartingIndexOfTest);
      // Note - no need to check for labeled data now, since we have no visitedPoints
      // => no labeled data
      System.out.println("First point selected: " + point);
      return point;
    }
    else {
      if (m_Verbose) {
	Iterator iter = visitedPoints.iterator();
	while(iter.hasNext()) {
	  System.out.println("In visitedPoints set: " + ((Integer) iter.next()).intValue());
	}
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
    ArrayList bestPoints = new ArrayList();

    for (int i=0; i<m_Instances.numInstances() && i<m_StartingIndexOfTest; i++) {
    // point should not belong to test set
      if (!visitedPoints.contains(new Integer(i))) {
	if (eliminationSet == null || !eliminationSet.contains(new Integer(i))) {
	  // point should not belong to visitedPoints
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
		//	      if (m_Verbose) {
		//		System.out.println("Max similarity of " + i + " from set is: " + maxSimilarityFromSet);
		//	      }
	      }
	    }
	    else {
	      double dist = m_metric.distance(inst, pointInSet);
	      if (dist < minDistanceFromSet) {
		minDistanceFromSet = dist;
		//	      if (m_Verbose) {
		//		System.out.println("Min distance of " + i + " from set is: " + minDistanceFromSet);
		//	      }
	      }
	    }
	  }
	  if (m_Verbose) {
	    System.out.println(i + " has sim: " + maxSimilarityFromSet + ", best: " + minSimilaritySoFar);
	  }
	  if (!m_objFunDecreasing) {
	    if (maxSimilarityFromSet == minSimilaritySoFar) {
	      minSimilaritySoFar = maxSimilarityFromSet;
	      bestPoints.add(new Integer(i));
	      if (m_Verbose) {
		System.out.println("Additional point added: " + i + " with similarity: " + minSimilaritySoFar);
	      }
	    }
	    else if (maxSimilarityFromSet < minSimilaritySoFar) {
	      minSimilaritySoFar = maxSimilarityFromSet;
	      bestPoints.clear();
	      bestPoints.add(new Integer(i));
	      if (m_Verbose) {
		System.out.println("Farthest point from set is: " + i + " with similarity: " + minSimilaritySoFar);
	      }
	    }
	  }
	  else {
	    if (minDistanceFromSet == maxDistanceSoFar) {
	      minDistanceFromSet = maxDistanceSoFar;
	      bestPoints.add(new Integer(i));
	      if (m_Verbose) {
		System.out.println("Additional point added: " + i + " with similarity: " + minSimilaritySoFar);
	      }
	    }
	    else if (minDistanceFromSet > maxDistanceSoFar) {
	      maxDistanceSoFar = minDistanceFromSet;
	      bestPoints.clear();
	      bestPoints.add(new Integer(i));
	      if (m_Verbose) {
		System.out.println("Farthest point from set is: " + i + " with distance: " + maxDistanceSoFar);
	      }
	    }
	  }
	}
      }
    }

    int bestPoint = -1;
    if (bestPoints.size() > 1) { // multiple points, get random from whole set
      Random random = new Random(m_randomSeed);
      bestPoint = random.nextInt(m_StartingIndexOfTest);
      while ((visitedPoints != null && visitedPoints.contains(new Integer(bestPoint))) ||
	     (eliminationSet != null && eliminationSet.contains(new Integer(bestPoint)))) {
	bestPoint = random.nextInt(m_StartingIndexOfTest);
      }
      System.out.println("Randomly selected " + bestPoint + " with similarity: " + minSimilaritySoFar);
    }
    else { // only 1 point, fine
      bestPoint = ((Integer)bestPoints.get(0)).intValue();
      System.out.println("Deterministically selected " + bestPoint + " with similarity: " + minSimilaritySoFar);
    }
    if (m_Verbose) {
      if (!m_objFunDecreasing) {
	System.out.println("Randomly selected " + bestPoint + " with similarity: " + minSimilaritySoFar);
      }
      else {
	System.out.println("Randomly selected " + bestPoint + " with similarity: " + maxDistanceSoFar);
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
    for (int i=0; i<m_Instances.numInstances() && i<m_StartingIndexOfTest; i++) {
      // bestPoint should not be a test point
      if (!visitedPoints.contains(new Integer(i))) {
	// bestPoint should not belong to visitedPoints
	Instance inst = m_Instances.instance(i);
	if (!m_objFunDecreasing) {
	  double sim = m_metric.similarity(inst, center);
	  if (sim > maxSimilarity) {
	    bestPoint = i;
	    maxSimilarity = sim;
	    if (m_Verbose) {
	      System.out.println("Nearest point is: " + bestPoint + " with sim: " + maxSimilarity);
	    }
	  }
	}
	else {
	  double dist = m_metric.distance(inst, center);
	  if (dist < minDistance) {
	    bestPoint = i;
	    minDistance = dist;
	    if (m_Verbose) {
	      System.out.println("Nearest point is: " + bestPoint + " with dist: " + minDistance);
	    }
	  }
	}
      }
    }
    return bestPoint;
  }


  /** Finds sum of instances (handles sparse and non-sparse)
   */

  protected Instance sumInstances(Instance inst1, Instance inst2) throws Exception {
    int numAttributes = inst1.numAttributes();
    if (inst2.numAttributes() != numAttributes) {
      throw new Exception ("Error!! inst1 and inst2 should have same number of attributes.");
    }
    if (m_Verbose) {
      //      System.out.println("Instance 1 is: " + inst1 + ", instance 2 is: " + inst2);
    }
    double weight1 = inst1.weight(), weight2 = inst2.weight();
    double [] values = new double[numAttributes];
    
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
      SparseInstance newInst = new SparseInstance(weight1+weight2, values);
      newInst.setDataset(m_Instances);
      if (m_Verbose) {
	//	System.out.println("Sum instance is: " + newInst);
      }
      return newInst;
    }
    else if (!(inst1 instanceof SparseInstance) && !(inst2 instanceof SparseInstance)){
      for (int i=0; i<numAttributes; i++) {
	values[i] = inst1.value(i) + inst2.value(i);
      }
    }
    else {
      throw new Exception ("Error!! inst1 and inst2 should be both of same type -- sparse or non-sparse");
    }
    Instance newInst = new Instance(weight1+weight2, values);
    newInst.setDataset(m_Instances);
    if (m_Verbose) {
      //      System.out.println("Sum instance is: " + newInst);
    }
    return newInst;
  }

  /** This function divides every attribute value in an instance by
   *  the instance weight -- useful to find the mean of a cluster in
   *  Euclidean space 
   *  @param inst Instance passed in for normalization (destructive update)
   */
  protected void normalizeByWeight(Instance inst) {
    double weight = inst.weight();
    if (m_Verbose) {
      //      System.out.println("Before weight normalization: " + inst);
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
    if (m_Verbose) {
      //      System.out.println("After weight normalization: " + inst);
    }
  }

  public int[] oldBestInstancesForActiveLearning(int numActive) throws Exception{
    
    int numInstances = m_Instances.numInstances();
    double [] scores = new double [numInstances];
    int numLabeledData = 0;
    if (m_SeedHash != null) {
      numLabeledData = m_SeedHash.size();
    }
    // Remember: order of data -- labeled, then unlabeled, then test

    for (int i=0; i<numLabeledData; i++) {
      scores[i] = -1;
    }

    for (int i=numLabeledData; i<numInstances; i++) {
      double score = 0, normalizer = 0;
      Instance inst = m_Instances.instance(i);
      double[] prob = new double[m_NumClusters];
      
      for (int j=0; j<m_NumClusters; j++) {
	if (!m_objFunDecreasing) {
	  double sim = m_metric.similarity(inst, m_ClusterCentroids.instance(j));
	  prob[j] = Math.exp(sim * m_Concentration); // P(x|h)
	}
	else {
	  double dist = m_metric.distance(inst, m_ClusterCentroids.instance(j));
	  prob[j] = Math.exp(-dist*dist * m_Concentration); // P(x|h)
	}
	normalizer += prob[j]; // P(x)/P(h) = Sum_h P(x|h) [uniform priors P(h)]
      }
      for (int j=0; j<m_NumClusters; j++) {
	prob[j] /= normalizer; // P(h|x) = P(x|h)*P(h)/P(x)
	score -= prob[j] * Math.log(prob[j]);
      }
      scores[i] = score * normalizer; // InfoGain = H(C|x).P(x) [with a constant factor of 1/P(h)]
    }

    System.out.println("NumInstances: "+ numInstances + ", starting index of unlabeled train: " + numLabeledData + ", starting index of test: " + m_StartingIndexOfTest);
    int [] indices = Utils.sort(scores);
    int [] mostConfused = new int [numActive];
    for (int i=0,num=0; i<numInstances && num<numActive; i++) {
      int index = numInstances-1-i;
      if ((indices[index]<m_StartingIndexOfTest) && (scores[indices[index]]!=-1)) { 
	// makes sure that labeled or test instances are not asked to be active labeled
	mostConfused[num] = (indices[index]);
	num++;
      }
    }
    for (int i=0; i<numActive; i++) {
      //      System.out.println("Value: " + scores[mostConfused[i]] + ", index: " + mostConfused[i]);
    }
    return mostConfused;
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
    return assignClusterToInstance(instance);
  }

  /**
   * Classifies the instance using the current clustering
   *
   * @param instance the instance to be assigned to a cluster
   * @return the number of the assigned cluster as an integer
   * if the class is enumerated, otherwise the predicted value
   * @exception Exception if instance could not be classified
   * successfully */

  public int assignClusterToInstance(Instance instance) throws Exception {
    int bestCluster = 0;
    double bestDistance = Double.POSITIVE_INFINITY;
    double bestSimilarity = Double.NEGATIVE_INFINITY;

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

    if (bestSimilarity == 0) {
      System.out.println("Note!! bestSimilarity is 0 for instance " + m_currIdx + ", assigned to cluster: " + bestCluster + " ... instance is: " + instance);
    }
    
    return bestCluster;
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


  /** Return the number of extra phase1 runs */
  public double getExtraPhase1RunFraction() {
    return m_ExtraPhase1RunFraction;
  }

  /** Set the number of extra phase1 runs */
  public void setExtraPhase1RunFraction(double w) {
    m_ExtraPhase1RunFraction = w;
  }


  /** Return the concentration */
  public double getConcentration() {
    return m_Concentration;
  }

  /** Set the concentration */
  public void setConcentration(double w) {
    m_Concentration = w;
  }


  /** Set the m_SeedHash */
  public void setSeedHash(HashMap seedhash) {
    m_SeedHash = seedhash;
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
    if (m_Verbose) {
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
    m_metricName = m_metric.getClass().getName();
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
   * Get the distance metric name
   *
   * @returns the name of the distance metric used
   */
  public String metricName () {
    return m_metricName;
  }

  /**
   * Set the seeding method.  Values other than
   * SEEDING_CONSTRAINED, or SEEDING_SEEDED will be ignored
   *
   * @param seedingMethod the seeding method to use
   */
  public void setSeedingMethod (SelectedTag seedingMethod)
  {
    if (seedingMethod.getTags() == TAGS_SEEDING) {
      if (m_Verbose) {
	System.out.println("Seeding method: " + seedingMethod.getSelectedTag().getReadable());
      }
      m_SeedingMethod = seedingMethod.getSelectedTag().getID();
    }
  }

  /**
   * Get the seeding method used. 
   *
   * @returns the seeding method 
   */
  public SelectedTag getSeedingMethod ()
  {
      return new SelectedTag(m_SeedingMethod, TAGS_SEEDING);
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
      if (m_Verbose) {
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
   * Set the distance metric
   *
   * @param met the distance metric that should be used
   */
  public void setMetricName (String metricName) {
    try {
      m_metricName = metricName;
      m_metric = (Metric) Class.forName(metricName).newInstance();
      m_objFunDecreasing = m_metric.isDistanceBased();
    } 
    catch (Exception e) {
      System.err.println("Error instantiating metric " + metricName);
    }
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


  /** Turn seeding on and off
   * @param seedable should seeding be done?
   */
  public boolean getSeedable() {
      return m_Seedable;
  }
    
  /** Read the seeds from a hastable, where every key is an instance and every value is:
   * the cluster assignment of that instance 
   * seedVector vector containing seeds
   */
  
  public void seedClusterer(HashMap seedHash) {
    if(m_Seedable) {
      setSeedHash(seedHash);
    }
  }
  
  /**
   * Computes the clusters from the cluster assignments, for external access
   * 
   * @exception Exception if clusters could not be computed successfully
   */    
  
  public ArrayList getIndexClusters() throws Exception {
    m_IndexClusters = new ArrayList();
    Cluster [] clusterArray = new Cluster[m_Instances.numInstances()];
    for (int i=0; i < m_Instances.numInstances(); i++) {
      if (m_ClusterAssignments[i]!=-1) {
	if (clusterArray[m_ClusterAssignments[i]] == null) {
	  clusterArray[m_ClusterAssignments[i]] = new Cluster();
	}
	clusterArray[m_ClusterAssignments[i]].add(new Integer(i), 1);
      }
    }
    
    for (int j =0; j< m_Instances.numInstances(); j++) 
      m_IndexClusters.add(clusterArray[j]);
    
    return m_IndexClusters;
  }

  /** Outputs the current clustering
   *
   * @exception Exception if something goes wrong
   */
  public void printIndexClusters() throws Exception {
    if (m_IndexClusters == null)
      throw new Exception ("Clusters were not created");

    for (int i = 0; i < m_IndexClusters.size(); i++) {
      Cluster cluster = (Cluster) m_IndexClusters.get(i);
      if (cluster == null) {
	//	System.out.println("Cluster " + i + " is null");
      }
      else {
	System.out.println ("Cluster " + i + " consists of " + cluster.size() + " elements");
	for (int j = 0; j < cluster.size(); j++) {
	  int idx = ((Integer) cluster.get(j)).intValue();
	  System.out.println("\t\t" + idx);
	}
      }
    }
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
   * Computes the final clusters from the cluster assignments, for external access
   * 
   * @exception Exception if clusters could not be computed successfully
   */    

  public ArrayList getClusters() throws Exception {
    m_FinalClusters = new ArrayList();
    Cluster [] clusterArray = new Cluster[m_NumClusters];

    for (int i=0; i < m_Instances.numInstances(); i++) {
	Instance inst = m_Instances.instance(i);
	if(clusterArray[m_ClusterAssignments[i]] == null)
	   clusterArray[m_ClusterAssignments[i]] = new Cluster();
	clusterArray[m_ClusterAssignments[i]].add(inst, 1);
    }

    for (int j =0; j< m_NumClusters; j++) 
      m_FinalClusters.add(clusterArray[j]);

    return m_FinalClusters;
  }


  public Enumeration listOptions () {
    return null;
  }

  /**
   * Gets the classifier specification string, which contains the class name of
   * the classifier and any options to the classifier
   *
   * @return the classifier string.
   */
  protected String getMetricSpec() {
    if (m_metric instanceof OptionHandler) {
      return m_metric.getClass().getName() + " "
	+ Utils.joinOptions(((OptionHandler)m_metric).getOptions());
    }
    return m_metric.getClass().getName();
  }
  
  public String [] getOptions ()  {
    String[] options = new String[80];
    int current = 0;

    options[current++] = "-N";
    options[current++] = "" + getNumClusters();
    options[current++] = "-R";
    options[current++] = "" + getRandomSeed();
    if (getSeedable()) {
      options[current++] = "-S";
      options[current++] = "" + getSeedingMethod().getSelectedTag().getID();
    }
    options[current++] = "-A";
    options[current++] = "" + getAlgorithm().getSelectedTag().getID();

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

    optionString = Utils.getOption('S', options);
    if (optionString.length() != 0) {
      setSeedingMethod(new SelectedTag(Integer.parseInt(optionString), TAGS_SEEDING));
    }
    else {
      setSeedable(false);
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
      if (m_Verbose) {
	System.out.println("Metric name: " + metricName + "\nMetric parameters: " + concatStringArray(metricSpec));
      }
      setMetric((LearnableMetric) LearnableMetric.forName(metricName, metricSpec));
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
   * return a string describing this clusterer
   *
   * @return a description of the clusterer as a string
   */
  public String toString() {
    StringBuffer temp = new StringBuffer();

    temp.append("\nkMeans\n======\n");
    temp.append("\nNumber of iterations: " + m_Iterations+"\n");

    temp.append("\nCluster centroids:\n");
    for (int i = 0; i < m_NumClusters; i++) {
      temp.append("\nCluster "+i+"\n\t");
      /*
        temp.append(m_ClusterCentroids.instance(i));
	for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
	if (m_ClusterCentroids.attribute(j).isNominal()) {
	temp.append(" "+m_ClusterCentroids.attribute(j).
	value((int)m_ClusterCentroids.instance(i).value(j)));
	} 
	else {
	temp.append(" "+m_ClusterCentroids.instance(i).value(j));
	}
	}
      */
    }
    temp.append("\n");
    return temp.toString();
  }


  /**
   * set the verbosity level of the clusterer
   * @param verbose messages on(true) or off (false)
   */
  public void setVerbose (boolean verbose) {
    m_Verbose = verbose;
  }

  /**
   * get the verbosity level of the clusterer
   * @return messages on(true) or off (false)
   */
  public boolean getVerbose () {
    return m_Verbose;
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
      ((LearnableMetric) m_metric).normalizeInstanceWeighted(inst);
    }
  }

  /** Normalizes the values of a normal Instance
   *
   * @author Sugato Basu
   * @param inst Instance to be normalized
   */

  public void normalizeInstance(Instance inst) throws Exception{
    double norm = 0;
    double values [] = inst.toDoubleArray();

    if (inst instanceof SparseInstance) {
      throw new Exception("Use normalizeSparseInstance function");
    }

    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	norm += values[i] * values[i];
      }
    }
    norm = Math.sqrt(norm);
    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	if (norm == 0) {
	  values[i]= 0;
	} else {
	  values[i] /= norm;
	}
      }
    }
    inst.setValueArray(values);
  }

  /** Normalizes the values of a SparseInstance
   *
   * @author Sugato Basu
   * @param inst SparseInstance to be normalized
   */

  public void normalizeSparseInstance(Instance inst) throws Exception{
    double norm=0;
    int length = inst.numValues();

    if (!(inst instanceof SparseInstance)) {
      throw new Exception("Use normalizeInstance function");
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
   * Main method for testing this class.
   *
   */

  public static void main (String[] args) {
    try {    
      String dataSet = new String("news");
      //String dataSet = new String("iris");
      if (dataSet.equals("iris")) {
	//////// Iris data
	String datafile = "/u/ml/software/weka-latest/data/iris.arff";
	
	// set up the data
	FileReader reader = new FileReader (datafile);
	Instances data = new Instances (reader);
	
	// Make the last attribute be the class 
	int theClass = data.numAttributes();
	data.setClassIndex(theClass-1); // starts with 0
	
	// Remove the class labels before clustering
	
	Instances clusterData = new Instances(data);
	clusterData.deleteClassAttribute();

	// #clusters = #classes
	int num_clusters = data.numClasses();
	
	// cluster with seeding      
	Instances seeds = new Instances(data,0,5);
	seeds.add(data.instance(50));
	seeds.add(data.instance(51));
	seeds.add(data.instance(52));
	seeds.add(data.instance(53));
	seeds.add(data.instance(54));
	
	seeds.add(data.instance(100));
	seeds.add(data.instance(101));
	seeds.add(data.instance(102));
	seeds.add(data.instance(103));
	seeds.add(data.instance(104));

	data.delete(104);
	data.delete(103);
	data.delete(102);
	data.delete(101);
	data.delete(100);
	data.delete(54);
	data.delete(53);
	data.delete(52);
	data.delete(51);
	data.delete(50);
	data.delete(4);
	data.delete(3);
	data.delete(2);
	data.delete(1);
	data.delete(0);	

	System.out.println("\nClustering the iris data with seeding, using seeded KMeans...\n");      
	WeightedEuclidean euclidean = new WeightedEuclidean();
	SeededKMeans kmeans = new SeededKMeans (euclidean);
	kmeans.resetClusterer();
	kmeans.setVerbose(false);
	kmeans.setSeedingMethod(new SelectedTag(SEEDING_SEEDED, TAGS_SEEDING));
	kmeans.setAlgorithm(new SelectedTag(ALGORITHM_SIMPLE, TAGS_ALGORITHM));
	euclidean.setExternal(false);
	euclidean.setTrainable(false);

	// phase 1 test
	kmeans.setSeedable(false);
	kmeans.buildClusterer(null, clusterData, theClass, data, 150);

	// phase 2 test
	//kmeans.setSeedable(true);
	//kmeans.buildClusterer(seeds, clusterData, theClass, data, 150);

	kmeans.getIndexClusters();
	kmeans.printIndexClusters();
	//	kmeans.setVerbose(true);
	kmeans.bestInstancesForActiveLearning(50);
      }
      else if (dataSet.equals("news")) {
	//////// Text data - 3000 documents
	String datafile = "/u/ml/data/CCSfiles/arffFromCCS/cmu-newsgroup-clean-1000_fromCCS.arff";
	System.out.println("\nClustering complete newsgroup data with seeding, using constrained KMeans...\n");      

	// set up the data
	FileReader reader = new FileReader (datafile);
	Instances data = new Instances (reader);
	System.out.println("Initial data has size: " + data.numInstances());
	
	// Make the last attribute be the class 
	int theClass = data.numAttributes();
	data.setClassIndex(theClass-1); // starts with 0
	int num_clusters = data.numClasses();
	
	// cluster with seeding      
        Instances seeds = new Instances(data, 0);
	/*
	seeds.add(data.instance(994));
	seeds.add(data.instance(1431));
	seeds.add(data.instance(1612));
	seeds.add(data.instance(1747));
	seeds.add(data.instance(2205));
	seeds.add(data.instance(2736));
	data.delete(2736);
	data.delete(2205);
	data.delete(1747);
	data.delete(1612);
	data.delete(1431);
	data.delete(994);

	seeds.add(data.instance(1000));
	seeds.add(data.instance(1001));
	seeds.add(data.instance(1002));
	seeds.add(data.instance(1003));
	seeds.add(data.instance(1004));
	seeds.add(data.instance(2000));
	seeds.add(data.instance(2001));
	seeds.add(data.instance(2002));
	seeds.add(data.instance(2003));
	seeds.add(data.instance(2004));

	//        System.out.println("Labeled data has size: " + seeds.numInstances() + ", number of attributes: " + data.numAttributes());

	data.delete(2004);
	data.delete(2003);
	data.delete(2002);
	data.delete(2001);
	data.delete(2000);
	data.delete(1004);
	data.delete(1003);
	data.delete(1002);
	data.delete(1001);
	data.delete(1000);
	data.delete(4);
	data.delete(3);
	data.delete(2);
	data.delete(1);
	data.delete(0);
	*/
	System.out.println("Unlabeled data has size: " + data.numInstances());
	
	// Remove the class labels before clustering
	
	Instances clusterData = new Instances(data);
	clusterData.deleteClassAttribute();

	WeightedDotP dotp = new WeightedDotP();
	dotp.setExternal(false);
	dotp.setTrainable(false);
	dotp.setLengthNormalized(false);
	SeededKMeans kmeans = new SeededKMeans(dotp);
	kmeans.setVerbose(false);
	kmeans.setSeedingMethod(new SelectedTag(SEEDING_SEEDED, TAGS_SEEDING));
	kmeans.setAlgorithm(new SelectedTag(ALGORITHM_SPHERICAL, TAGS_ALGORITHM));
	kmeans.setNumClusters(3);

	// phase 1 test
	kmeans.setSeedable(false);
	kmeans.buildClusterer(null, clusterData, theClass, data, data.numInstances());

	// phase 2 test
	//kmeans.setSeedable(true);
	//kmeans.buildClusterer(seeds, clusterData, theClass, data, 3000);

	kmeans.getIndexClusters();
	kmeans.printIndexClusters();
	//	kmeans.setVerbose(true);
	//kmeans.bestInstancesForActiveLearning(50);
      
//  	// cluster with seeding for small newsgroup
//  	seeds = new Instances(data, 0, 3);
	
//  	seeds.add(data.instance(100)); 
//  	seeds.add(data.instance(101));
//  	seeds.add(data.instance(102));
//  	seeds.add(data.instance(200));
//  	seeds.add(data.instance(201));
//  	seeds.add(data.instance(202));
//  	seeds.add(data.instance(300));
//  	seeds.add(data.instance(301));
//  	seeds.add(data.instance(302));
//  	seeds.add(data.instance(400));
//  	seeds.add(data.instance(401));
//  	seeds.add(data.instance(402));
//  	seeds.add(data.instance(500));
//  	seeds.add(data.instance(501));
//  	seeds.add(data.instance(502));
//  	seeds.add(data.instance(600));
//  	seeds.add(data.instance(601));
//  	seeds.add(data.instance(602));
//  	seeds.add(data.instance(700));
//  	seeds.add(data.instance(701));
//  	seeds.add(data.instance(702));
//  	seeds.add(data.instance(800));
//  	seeds.add(data.instance(801));
//  	seeds.add(data.instance(802));
//  	seeds.add(data.instance(900));
//  	seeds.add(data.instance(901));      
//  	seeds.add(data.instance(902));
//  	seeds.add(data.instance(1000)); 
//  	seeds.add(data.instance(1001));
//  	seeds.add(data.instance(1002));
//  	seeds.add(data.instance(1100)); 
//  	seeds.add(data.instance(1101));
//  	seeds.add(data.instance(1102));
//  	seeds.add(data.instance(1200));
//  	seeds.add(data.instance(1201));
//  	seeds.add(data.instance(1202));
//  	seeds.add(data.instance(1300));
//  	seeds.add(data.instance(1301));
//  	seeds.add(data.instance(1302));
//  	seeds.add(data.instance(1400));
//  	seeds.add(data.instance(1401));
//  	seeds.add(data.instance(1402));
//  	seeds.add(data.instance(1500));
//  	seeds.add(data.instance(1501));
//  	seeds.add(data.instance(1502));
//  	seeds.add(data.instance(1600));
//  	seeds.add(data.instance(1601));
//  	seeds.add(data.instance(1602));
//  	seeds.add(data.instance(1700));
//  	seeds.add(data.instance(1701));
//  	seeds.add(data.instance(1702));
//  	seeds.add(data.instance(1800));
//  	seeds.add(data.instance(1801));
//  	seeds.add(data.instance(1802));
//  	seeds.add(data.instance(1900));
//  	seeds.add(data.instance(1901));
//  	seeds.add(data.instance(1902));
	
//  	System.out.println("Labeled data has size: " + seeds.numInstances() + ", number of attributes: " + data.numAttributes());
	
//  	data.delete(1902);
//  	data.delete(1901);
//  	data.delete(1900);
//  	data.delete(1802);
//  	data.delete(1801);
//  	data.delete(1800);
//  	data.delete(1702);
//  	data.delete(1701);
//  	data.delete(1700);
//  	data.delete(1602);
//  	data.delete(1601);
//  	data.delete(1600);
//  	data.delete(1502);
//  	data.delete(1501);
//  	data.delete(1500);
//  	data.delete(1402);
//  	data.delete(1401);
//  	data.delete(1400);
//  	data.delete(1302);
//  	data.delete(1301);
//  	data.delete(1300);
//  	data.delete(1202);
//  	data.delete(1201);
//  	data.delete(1200);
//  	data.delete(1102);
//  	data.delete(1101);
//  	data.delete(1100);
//  	data.delete(1002);
//  	data.delete(1001);
//  	data.delete(1000);
//  	data.delete(902);
//  	data.delete(901);
//  	data.delete(900);
//  	data.delete(802);
//  	data.delete(801);
//  	data.delete(800);
//  	data.delete(702);
//  	data.delete(701);
//  	data.delete(700);
//  	data.delete(602);
//  	data.delete(601);
//  	data.delete(600);
//  	data.delete(502);
//  	data.delete(501);
//  	data.delete(500);
//  	data.delete(402);
//  	data.delete(401);
//  	data.delete(400);
//  	data.delete(302);
//  	data.delete(301);
//  	data.delete(300);
//  	data.delete(202);
//  	data.delete(201);
//  	data.delete(200);
//  	data.delete(102);
//  	data.delete(101);
//  	data.delete(100);
//  	data.delete(2);
//  	data.delete(1);
//  	data.delete(0);
      }
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }
}
