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
 *    MPCKMeans.java
 *    Copyright (C) 2003 Sugato Basu and Misha Bilenko
 *
 */
package weka.clusterers;

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.core.metrics.*;
import  weka.filters.Filter;
import  weka.filters.unsupervised.attribute.Remove;
import  Jama.Matrix;
import  Jama.EigenvalueDecomposition;
import  weka.clusterers.assigners.*;
import  weka.clusterers.regularizers.*;
import  weka.clusterers.initializers.*;
import  weka.clusterers.metriclearners.*;



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
 * -M <metric-class> <br>
 * Specifies the name of the distance metric class that should be used
 * 
 * @author Sugato Basu(sugato@cs.utexas.edu) and Misha Bilenko (mbilenko@cs.utexas.edu)
 * @see Clusterer
 * @see OptionHandler
 */
public class MPCKMeans extends Clusterer implements OptionHandler,SemiSupClusterer {

  /** Name of clusterer */
  String m_name = "MPCKMeans";

  /** holds the instances in the clusters */
  protected ArrayList m_Clusters = null;

  /** holds the instance indices in the clusters */
  protected HashSet[] m_IndexClusters = null;
  
  /** holds the ([instance pair] -> [type of constraint]) mapping,
      where the hashed value stores the type of link but the instance
      pair does not hold the type of constraint - it holds (instanceIdx1,
      instanceIdx2, DONT_CARE_LINK). This is done to make lookup easier
      in future 
  */
  protected HashMap m_ConstraintsHash = null;
  public HashMap getConstraintsHash() {
    return m_ConstraintsHash;
  }

  /** stores the ([instanceIdx] -> [ArrayList of constraints])
      mapping, where the arraylist contains the constraints in which
      instanceIdx is involved. Note that the instance pairs stored in
      the Arraylist have the actual link type.  
  */
  protected HashMap m_instanceConstraintHash = null; 
  public HashMap getInstanceConstraintsHash() {
    return m_instanceConstraintHash;
  }

  public void setInstanceConstraintsHash(HashMap instanceConstraintHash) {
    m_instanceConstraintHash = instanceConstraintHash;
  }
  
  /** holds the points involved in the constraints */
  protected HashSet m_SeedHash = null;

  /** Access */
  public HashSet getSeedHash () {
    return m_SeedHash;
  }
  
  /** weight to be given to each constraint */
  protected double m_CLweight = 1;

  /** weight to be given to each constraint */
  protected double m_MLweight = 1;

  /** should constraints from transitive closure be added? */
  protected boolean m_useTransitiveConstraints = true;   

  /** is it an offline metric (BarHillelMetric or XingMetric)? */
  protected boolean m_isOfflineMetric;

  public boolean getIsOfflineMetric () {
    return m_isOfflineMetric;
  }
  
  /** the maximum distance between cannot-link constraints */
  protected double m_MaxCannotLinkDistance = 0;

  /** the min similarity between cannot-link constraints */
  protected double m_MaxCannotLinkSimilarity = 0;

  /** the maximum distance between cannot-link constraints */
  protected double m_maxCLPenalties[] = null;
  public Instance m_maxCLPoints[][] = null;
  public Instance m_maxCLDiffInstances[] = null;

  /** verbose? */
  protected boolean m_verbose = false;

  /** distance Metric */
  protected LearnableMetric m_metric = new WeightedEuclidean();
  protected MPCKMeansMetricLearner m_metricLearner = new WEuclideanLearner();

  /** Individual metrics for each cluster can be used */
  protected boolean m_useMultipleMetrics = false;
  protected LearnableMetric [] m_metrics = null;
  protected MPCKMeansMetricLearner [] m_metricLearners = null;

  /** Relative importance of the log-term for the weights in the objective function */
  protected double m_logTermWeight = 0.01;


  /** Regularization for weights */
  protected boolean m_regularize = false; 
  protected double m_regularizerTermWeight = 0.001;
  

  /** We will hash log terms to avoid recomputing every time TODO:  implement for Euclidean*/
  protected double[] m_logTerms = null; 

  /** has the metric has been constructed?  a fix for multiple buildClusterer's */
  protected boolean m_metricBuilt = false;

  /** indicates whether instances are sparse */
  protected  boolean m_isSparseInstance = false;

  /** Is the objective function increasing or decreasing?  Depends on type
   * of metric used:  for similarity-based metric, increasing, for distance-based - decreasing */
  protected boolean m_objFunDecreasing = true;

  /** Seedable or not (true by default) */
  protected boolean m_Seedable = true;

  /** Possible metric training */
  public static final int TRAINING_NONE = 1;
  public static final int TRAINING_EXTERNAL = 2;
  public static final int TRAINING_INTERNAL = 4;
  public static final Tag[] TAGS_TRAINING = {
    new Tag(TRAINING_NONE, "None"),
    new Tag(TRAINING_EXTERNAL, "External"),
    new Tag(TRAINING_INTERNAL, "Internal")};
  
  protected int m_Trainable = TRAINING_INTERNAL;

  /** keep track of the number of iterations completed before convergence
   */
  protected int m_Iterations = 0;

    /** number of constraint violations
     */
  protected int m_numViolations = 0;


  /** keep track of the number of iterations when no points were moved */
  protected int m_numBlankIterations = 0;

  /** the maximum number of iterations */
  protected int m_maxIterations = Integer.MAX_VALUE;

  /** the maximum number of iterations with no points moved */
  protected int m_maxBlankIterations = 20;
  
  /** min difference of objective function values for convergence*/
  protected double m_ObjFunConvergenceDifference = 1e-5;

  /** value of current objective function */
  protected double m_Objective = Double.MAX_VALUE;

  /** value of last objective function */
  protected double m_OldObjective;

  /** Variables to track components of the objective function */
  protected double m_objVariance;
  protected double m_objCannotLinks;
  protected double m_objMustLinks;
  protected double m_objNormalizer;
  protected double m_objRegularizer;
  /** Variable to track the contribution of the currently considered point */
  protected double m_objVarianceCurrPoint;
  protected double m_objCannotLinksCurrPoint;
  protected double m_objMustLinksCurrPoint;
  protected double m_objNormalizerCurrPoint;

  protected double m_objVarianceCurrPointBest;
  protected double m_objCannotLinksCurrPointBest;
  protected double m_objMustLinksCurrPointBest;
  protected double m_objNormalizerCurrPointBest;

  /** returns objective function */
  public double objectiveFunction() {
    return m_Objective;
  }

  /**
   * training instances with labels
   */
  protected Instances m_TotalTrainWithLabels;
  public Instances getTotalTrainWithLabels() {
    return m_TotalTrainWithLabels;
  }
  public void setTotalTrainWithLabels(Instances inst) {
    m_TotalTrainWithLabels = inst;
  }

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
  protected int m_NumClusters = -1;

  /**
   * holds the cluster centroids
   */
  protected Instances m_ClusterCentroids;
  /** Accessor */ 
  public Instances getClusterCentroids() {
    return m_ClusterCentroids;
  } 

  public void setClusterCentroids(Instances centroids) {
    m_ClusterCentroids = centroids;
  } 
  
  /**
   * temporary variable holding cluster assignments while iterating
   */
  protected int [] m_ClusterAssignments;
  public int[] getClusterAssignments() {
    return m_ClusterAssignments;
  } 
  public void setClusterAssignments(int [] clusterAssignments) {
    m_ClusterAssignments = clusterAssignments;
  } 

  protected String m_ClusterAssignmentsOutputFile;
  public String getClusterAssignmentsOutputFile() {
    return m_ClusterAssignmentsOutputFile;
  } 
  public void setClusterAssignmentsOutputFile(String file) {
    m_ClusterAssignmentsOutputFile = file;
  } 

  protected String m_ConstraintIncoherenceFile;
  public String getConstraintIncoherenceFile() {
    return m_ConstraintIncoherenceFile;
  } 
  public void setConstraintIncoherenceFile(String file) {
    m_ConstraintIncoherenceFile = file;
  } 


  /**
   * holds the random Seed, useful for randomPerturbInit
   */
  protected int m_RandomSeed = 42;

  /**
   * holds the random number generator used in various parts of the code
   */
  protected Random m_RandomNumberGenerator = null;

  /** Define possible assignment strategies */
  protected MPCKMeansAssigner m_Assigner = new SimpleAssigner(this);

  /** Define possible initialization strategies */
  //  protected MPCKMeansInitializer m_Initializer = new RandomPerturbInitializer(this);
  protected MPCKMeansInitializer m_Initializer = new WeightedFFNeighborhoodInit(this);


  /** Access */
  public Random getRandomNumberGenerator() {
    return m_RandomNumberGenerator;
  }
  
  /* Constructor */
  public MPCKMeans() {
  }

  /* Constructor */
  public MPCKMeans(LearnableMetric metric) {
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
   * @param numClusters number of clusters to create
   * @exception Exception if something goes wrong.
   */
  public void buildClusterer(Instances data, int numClusters) throws Exception {
    m_NumClusters = numClusters;
    System.out.println("Creating " + m_NumClusters + " clusters");
    m_Initializer.setNumClusters(m_NumClusters);
    
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
  public void buildClusterer (Instances labeledData, Instances unlabeledData,
			      int classIndex, int numClusters,
			      int startingIndexOfTest) throws Exception {
    // Dummy function
    throw new Exception ("Not implemented for MPCKMeans, only here for "
			 + "compatibility to SemiSupClusterer interface");
  }

  /**
   * Clusters unlabeledData and labeledData (with labels removed),
   * using constraints in labeledPairs to initialize
   *
   * @param labeledPairs labeled pairs to be used to initialize
   * @param unlabeledData unlabeled instances
   * @param labeledData labeled instances
   * @param numClusters number of clusters
   * @param startingIndexOfTest starting index of test set in unlabeled data
   * @exception Exception if something goes wrong.  */
  public void buildClusterer(ArrayList labeledPairs, Instances unlabeledData,
			     Instances labeledData, int numClusters,
			     int startingIndexOfTest) throws Exception {
    m_TotalTrainWithLabels = labeledData;

    if (labeledPairs != null) {
      m_SeedHash = new HashSet((int) (unlabeledData.numInstances()/0.75 + 10)) ;
      m_ConstraintsHash = new HashMap();
      m_instanceConstraintHash = new HashMap();

      for (int i = 0; i < labeledPairs.size(); i++) {
	InstancePair pair = (InstancePair) labeledPairs.get(i);	
	Integer firstInt = new Integer(pair.first);
	Integer secondInt = new Integer(pair.second);

	// for first point 
	if(!m_SeedHash.contains(firstInt)) { // add instances with constraints to seedHash
	  if (m_verbose) {
	    System.out.println("Adding " + firstInt + " to seedHash");
	  }
	  m_SeedHash.add(firstInt);
	}
	
	// for second point 
	if(!m_SeedHash.contains(secondInt)) {
	  m_SeedHash.add(secondInt);
	  if (m_verbose) {
	    System.out.println("Adding " + secondInt + " to seedHash");
	  }
	}
	if (pair.first >= pair.second) {
	  throw new Exception("Ordering reversed - something wrong!!");
	} else {
	  InstancePair newPair = null;
	  newPair = new InstancePair(pair.first, pair.second, InstancePair.DONT_CARE_LINK);
	  m_ConstraintsHash.put(newPair, new Integer(pair.linkType)); // WLOG first < second
	  if (m_verbose) {
	    System.out.println("Adding constraint (" + pair.first +","+pair.second+"), " + pair.linkType);
	  }
	  
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
    }

    m_StartingIndexOfTest = startingIndexOfTest;
    if (m_verbose) {
      System.out.println("Starting index of test: " + m_StartingIndexOfTest);
    }

    // learn metric using labeled data,
    // then cluster both the labeled and unlabeled data
    System.out.println("Initializing metric: " + m_metric);
    m_metric.buildMetric(unlabeledData);
    m_metricBuilt = true;
    m_metricLearner.setMetric(m_metric);
    m_metricLearner.setClusterer(this);

    // normalize all data for SPKMeans
    if (m_metric.doesNormalizeData()) {
      for (int i=0; i<unlabeledData.numInstances(); i++) {
	m_metric.normalizeInstanceWeighted(unlabeledData.instance(i));
      }
    }

    // either create a new metric if multiple metrics,
    // or just point them all to m_metric
    m_metrics = new LearnableMetric[numClusters];
    m_metricLearners = new MPCKMeansMetricLearner[numClusters];
    for (int i = 0; i < m_metrics.length; i++) {
      if (m_useMultipleMetrics) {
	m_metrics[i] = (LearnableMetric) m_metric.clone();
	m_metricLearners[i] = (MPCKMeansMetricLearner) m_metricLearner.clone();
	m_metricLearners[i].setMetric(m_metrics[i]);
	m_metricLearners[i].setClusterer(this);
      } else { 
	m_metrics[i] = m_metric;
	m_metricLearners[i] = m_metricLearner;
      } 
    } 
    buildClusterer(unlabeledData, numClusters);
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
    System.out.println("ML weight=" + m_MLweight);
    System.out.println("CL weight= " + m_CLweight);
    System.out.println("LOG term weight=" + m_logTermWeight);
    System.out.println("Regularizer weight= " + m_regularizerTermWeight);
    m_RandomNumberGenerator = new Random(m_RandomSeed);

    if (m_metric instanceof OfflineLearnableMetric) {
      m_isOfflineMetric = true;
    } else {
      m_isOfflineMetric = false;
    }

    // Don't rebuild the metric if it was already trained
    if (!m_metricBuilt) {
      m_metric.buildMetric(data);
      m_metricBuilt = true;
      m_metricLearner.setMetric(m_metric);
      m_metricLearner.setClusterer(this);
      
      m_metrics = new LearnableMetric[m_NumClusters];
      m_metricLearners = new MPCKMeansMetricLearner[m_NumClusters];
      for (int i = 0; i < m_metrics.length; i++) {
	if (m_useMultipleMetrics) {
	  m_metrics[i] = (LearnableMetric) m_metric.clone();
	  m_metricLearners[i] = (MPCKMeansMetricLearner) m_metricLearner.clone();
	  m_metricLearners[i].setMetric(m_metrics[i]);
	  m_metricLearners[i].setClusterer(this); 
	} else { 
	  m_metrics[i] = m_metric;
	  m_metricLearners[i] = m_metricLearner;
	} 
      } 
    }

    setInstances(data);
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);
    m_ClusterAssignments = new int [m_Instances.numInstances()];

    if (m_Instances.checkForNominalAttributes() &&
	m_Instances.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle nominal attributes\n");
    }

    m_ClusterCentroids = m_Initializer.initialize();

    // if all instances are smoothed by the metric, the centroids
    // need to be smoothed too (note that this is independent of
    // centroid smoothing performed by K-Means)
    if (m_metric instanceof InstanceConverter) {
      System.out.println("Converting centroids...");
      Instances convertedCentroids = new Instances(m_ClusterCentroids, m_NumClusters);
      for (int i = 0; i < m_ClusterCentroids.numInstances(); i++) {
	Instance centroid = m_ClusterCentroids.instance(i); 
	convertedCentroids.add(((InstanceConverter)m_metric).convertInstance(centroid));
      }

      m_ClusterCentroids.delete();
      for (int i = 0; i < convertedCentroids.numInstances(); i++) {
	m_ClusterCentroids.add(convertedCentroids.instance(i));
      }
    } 
    
    System.out.println("Done initializing clustering ...");
    getIndexClusters();

    if (m_verbose && m_Seedable) {
      printIndexClusters();
      for (int i=0; i<m_NumClusters; i++) {
	System.out.println("Centroid " + i + ": " + m_ClusterCentroids.instance(i));
      }
    }

    // Some extra work for smoothing metrics
    if (m_metric instanceof SmoothingMetric &&
	((SmoothingMetric) m_metric).getUseSmoothing()) { 

      SmoothingMetric smoothingMetric = (SmoothingMetric) m_metric;
      Instances smoothedCentroids = new Instances(m_Instances, m_NumClusters);
      
      for (int i = 0; i < m_ClusterCentroids.numInstances(); i++) {
	Instance smoothedCentroid =
	  smoothingMetric.smoothInstance(m_ClusterCentroids.instance(i)); 
	smoothedCentroids.add(smoothedCentroid);
      }
      m_ClusterCentroids = smoothedCentroids;

      updateSmoothingMetrics();       
    }

    runKMeans();
  }

  protected void updateSmoothingMetrics() {
    if (m_useMultipleMetrics) {
      for (int i = 0; i < m_NumClusters; i++) { 
	((SmoothingMetric)m_metrics[i]).updateAlpha();
      }
    } else {
      ((SmoothingMetric)m_metric).updateAlpha();
    }
  } 


  /**
   * Reset all values that have been learned
   */
  public void resetClusterer()  throws Exception{
    m_metric.resetMetric();
    if (m_useMultipleMetrics) {
      for (int i = 0; i < m_metrics.length; i++) {
	m_metrics[i].resetMetric();
      }
    }
    
    m_SeedHash = null;
    m_ConstraintsHash = null;
    m_instanceConstraintHash = null;
  }


  /** Turn seeding on and off
   * @param seedable should seeding be done?
   */
  public void setSeedable(boolean seedable) {
    m_Seedable = seedable;
  }

  /** Turn metric learning on and off
   * @param trainable should metric learning be done?
   */
  public void setTrainable(SelectedTag trainable) {
    if (trainable.getTags() == TAGS_TRAINING) {
      if (m_verbose) {
	System.out.println("Trainable: " + trainable.getSelectedTag().getReadable());
      }
      m_Trainable = trainable.getSelectedTag().getID();
    }
  }


  /** Is seeding performed?
   * @return is seeding being done?
   */
  public boolean getSeedable() {
    return m_Seedable;
  }

  /** Is metric learning performed?
   * @return is metric learning being done?
   */
  public SelectedTag getTrainable() {
    return new SelectedTag(m_Trainable, TAGS_TRAINING);
  }

  
  /**
   * We can have clusterers that don't utilize seeding
   */
  public boolean seedable() {
    return m_Seedable;
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
	  Instance inst = m_TotalTrainWithLabels.instance(idx);
	  if (m_TotalTrainWithLabels.classIndex() >= 0) {
	    System.out.println("\t\t" + idx + ":" + inst.classAttribute().value((int) inst.classValue()));
	  }
	}
      }
    }
  }


  /** E-step of the KMeans clustering algorithm -- find best cluster
   * assignments. Returns the number of points moved in this step 
   */
  protected int findBestAssignments() throws Exception {
    int moved = 0;
    double distance = 0;
    m_Objective = 0;
    m_objVariance = 0;
    m_objCannotLinks = 0;
    m_objMustLinks = 0;
    m_objNormalizer = 0;

    // Initialize the regularizer and normalizer hashes
    InitNormalizerRegularizer();
    
    if (m_isOfflineMetric) {
      moved = assignAllInstancesToClusters();
    } else {
      moved = assignPoints();
    }
    if (m_verbose) { 
      System.out.println("  " + moved + " points moved in this E-step");
    }
    return moved; 
  }

  /** Initialize m_logTerms and m_regularizerTerms */
  protected void InitNormalizerRegularizer() { 
    m_logTerms = new double[m_NumClusters];
    m_objRegularizer = 0;

    if (m_useMultipleMetrics) {
      for (int i = 0; i < m_NumClusters; i++) {
	m_logTerms[i] = m_logTermWeight * m_metrics[i].getNormalizer(); 

	if (m_regularize) {
	  m_objRegularizer += m_regularizerTermWeight * m_metrics[i].regularizer(); 
	}
      } 
    } else {  // we fill the logTerms with the log(det) of the only weight matrix
      m_logTerms[0] = m_logTermWeight * m_metric.getNormalizer();
      for (int i = 1; i < m_logTerms.length; i++) {
	m_logTerms[i] = m_logTerms[0];
      } 
      if (m_regularize) {
	m_objRegularizer = m_regularizerTermWeight * m_metric.regularizer(); 
      }
    }
  }

  
  /** Decides which assignment strategy to use based on argument passed in */
  int assignPoints() throws Exception {
    int moved = 0;

    moved = m_Assigner.assign();
    m_Objective = m_objVariance + m_objMustLinks
      + m_objCannotLinks + m_objNormalizer - m_objRegularizer;
    if (m_verbose) {
      System.out.println((float)m_Objective + " - Objective function (incomplete) after assignment");
      System.out.println("\tvar=" + ((float)m_objVariance)
			 + "\tC=" + ((float)m_objCannotLinks)
			 + "\tM=" + ((float)m_objMustLinks)
			 + "\tLOG=" + ((float)m_objNormalizer) 
			 + "\tREG=" + ((float)m_objRegularizer));
    }
    // TODO:  add a m_fast switch and put the following line inside it.
    //    calculateObjectiveFunction();
           
    return moved;
  }


  /**
   * Classifies the instance using the current clustering, considering constraints
   *
   * @param instance the instance to be assigned to a cluster
   * @return the number of the assigned cluster as an integer if the
   * class is enumerated, otherwise the predicted value
   * @exception Exception if instance could not be classified
   * successfully 
   */
  public int assignInstanceToClusterWithConstraints(int instIdx) throws Exception {
    int bestCluster = 0;
    double lowestPenalty = Double.MAX_VALUE;
    int moved = 0;

    // try each cluster and find one with lowest penalty
    for (int i = 0; i < m_NumClusters; i++) {
      double penalty = penaltyForInstance(instIdx, i);

      if (penalty < lowestPenalty) {
	lowestPenalty = penalty;
	bestCluster = i;
	m_objVarianceCurrPointBest = m_objVarianceCurrPoint;
	m_objNormalizerCurrPointBest = m_objNormalizerCurrPoint;
	m_objMustLinksCurrPointBest = m_objMustLinksCurrPoint;
	m_objCannotLinksCurrPointBest = m_objCannotLinksCurrPoint;
      }
    }
    
    m_objVariance += m_objVarianceCurrPointBest;
    m_objNormalizer += m_objNormalizerCurrPointBest;
    m_objMustLinks += m_objMustLinksCurrPointBest;
    m_objCannotLinks += m_objCannotLinksCurrPointBest;

    if (m_ClusterAssignments[instIdx] != bestCluster) {
      if (m_ClusterAssignments[instIdx] >= 0 && m_ClusterAssignments[instIdx] < m_NumClusters) {
	//if (m_verbose) {
	System.out.println("Moving instance " + instIdx + " from cluster "
			   + m_ClusterAssignments[instIdx] + " to cluster " + bestCluster
			   + " penalty:" + ((float)penaltyForInstance(instIdx, m_ClusterAssignments[instIdx]))
			   + "=>" + ((float)lowestPenalty)); 
      }
      moved = 1;
      m_ClusterAssignments[instIdx] = bestCluster; 
    }

    if (m_verbose) {
      System.out.println("Assigning instance " + instIdx + " to cluster "
			 + bestCluster);
    }

    return moved;
  }

  /** Delegate the distance calculation to the method appropriate for the current metric
   */
  public double penaltyForInstance(int instIdx, int centroidIdx) throws Exception {
    m_objVarianceCurrPoint = 0;
    m_objCannotLinksCurrPoint = 0;
    m_objMustLinksCurrPoint = 0;
    m_objNormalizerCurrPoint = 0;
    int violatedConstraints = 0; 

    // variance contribution
    Instance instance = m_Instances.instance(instIdx);
    Instance centroid =  m_ClusterCentroids.instance(centroidIdx);

    m_objVarianceCurrPoint = m_metrics[centroidIdx].penalty(instance, centroid);

    // regularizer and normalizer contribution
    if (m_Trainable == TRAINING_INTERNAL) {
      m_objNormalizerCurrPoint = -m_logTerms[centroidIdx]; 
    }

    // only add the constraints if seedable or constrained
    //    if (m_Seedable || (m_Trainable != TRAINING_NONE)) {   

    // Sugato: replacing, in order to be able to run MKMeans (no
    // constraint violation, only metric learning)
    if (m_Seedable) {
      Object list =  m_instanceConstraintHash.get(new Integer(instIdx));
      if (list != null) {   // there are constraints associated with this instance
	ArrayList constraintList = (ArrayList) list;
	for (int i = 0; i < constraintList.size(); i++) {
	  InstancePair pair = (InstancePair) constraintList.get(i);
	  int firstIdx = pair.first;
	  int secondIdx = pair.second;

	  Instance instance1 = m_Instances.instance(firstIdx);
	  Instance instance2 = m_Instances.instance(secondIdx);
	  int otherIdx = (firstIdx == instIdx) ? 
	    m_ClusterAssignments[secondIdx] : m_ClusterAssignments[firstIdx];
	  
	  // check whether the constraint is violated
	  if (otherIdx != -1 && otherIdx < m_NumClusters) { 
	    if (otherIdx != centroidIdx && 
		pair.linkType == InstancePair.MUST_LINK) {
	      violatedConstraints++; 
	      // split penalty in half between the two involved clusters
	      if (m_useMultipleMetrics) {  
		double penalty1 = m_metrics[otherIdx].penaltySymmetric(instance1, instance2);
		double penalty2 = m_metrics[centroidIdx].penaltySymmetric(instance1, instance2);
		m_objMustLinksCurrPoint += 0.5 * m_MLweight * (penalty1 + penalty2);
	      } else {
		double penalty = m_metric.penaltySymmetric(instance1, instance2);
		m_objMustLinksCurrPoint += m_MLweight * penalty;
	      }
	    } else if (otherIdx == centroidIdx &&
		       pair.linkType == InstancePair.CANNOT_LINK) {
	      violatedConstraints++; 
	      double penalty = m_metrics[centroidIdx].penaltySymmetric(instance1, instance2);
	      m_objCannotLinksCurrPoint +=  m_CLweight *
		(m_maxCLPenalties[centroidIdx] - penalty);
	      if (m_maxCLPenalties[centroidIdx] - penalty < 0) {
		System.out.println("***NEGATIVE*** penalty: " + penalty + " for CL constraint"); 
	      }
	    }
	  }
	}
      }
    }

    double total = m_objVarianceCurrPoint  + m_objCannotLinksCurrPoint 
      + m_objMustLinksCurrPoint + m_objNormalizerCurrPoint;
    if(m_verbose) {
      System.out.println("Final penalty for instance " + instIdx + " and centroid "
			 + centroidIdx + " is: " + total);
    }
    return total;
  }
  

  /** M-step of the KMeans clustering algorithm -- updates cluster centroids
   */
  protected void updateClusterCentroids() throws Exception {
    Instances [] tempI = new Instances[m_NumClusters];
    Instances tempCentroids = m_ClusterCentroids;
    Instances tempNewCentroids = new Instances(m_Instances, m_NumClusters); 
    m_ClusterCentroids = new Instances(m_Instances, m_NumClusters);

    // tempI[i] stores the cluster instances for cluster i
    for (int i = 0; i < m_NumClusters; i++) {
      tempI[i] = new Instances(m_Instances, 0); 
    }
    for (int i = 0; i < m_Instances.numInstances(); i++) {
      tempI[m_ClusterAssignments[i]].add(m_Instances.instance(i));
    }

    // Calculates cluster centroids
    for (int i = 0; i < m_NumClusters; i++) {
      double [] values = new double[m_Instances.numAttributes()];
      Instance centroid = null;
      
      if (m_isSparseInstance) { // uses fast meanOrMode
	values = ClusterUtils.meanOrMode(tempI[i]);
	centroid = new SparseInstance(1.0, values);
      } else { // non-sparse, go through each attribute
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  values[j] = tempI[i].meanOrMode(j); // uses usual meanOrMode
	}
	centroid = new Instance(1.0, values);
      }
      
//        // debugging:  compare  previous centroid w/current:
//        double w = 0; 
//        for (int j = 0; j < m_Instances.numAttributes(); j++)  w += values[j] * values[j];
//        double w1 = 0; 
//        for (int j = 0; j < m_Instances.numAttributes(); j++)  w1 += tempCentroids.instance(i).value(j) * tempCentroids.instance(i).value(j);
     

//        System.out.println("\tOldCentroid=" + w1);
//        System.out.println("\tNewCentroid=" + w); 
//        double prevObj = 0, currObj = 0;
//        for (int j = 0; j < tempI[i].numInstances(); j++) {
//  	Instance instance = tempI[i].instance(j);
//  	double prevPen = m_metrics[i].penalty(instance, tempCentroids.instance(i));
//  	double currPen = m_metrics[i].penalty(instance, centroid);
//  	prevObj += prevPen;
//  	currObj += currPen; 
//  	//System.out.println("\t\t" + j + " " + prevPen + " -> " + currPen + "\t" + prevObj + " -> " + currObj); 
//        }
//        // dump instances out if there is a problem.
//        System.out.println("\t\t" + prevObj + " -> " + currObj); 
//        if (currObj > prevObj) {

//  	PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream("/tmp/INST.arff")), true);
//  	out.println(new Instances(tempI[i], 0));
//  	out.println(centroid);
//  	out.println(tempCentroids.instance(i)); 
//  	for (int j = 0; j < tempI[i].numInstances(); j++) {
//  	  out.println(tempI[i].instance(j));
//  	}
//  	out.close();
//  	System.out.println("  Updated cluster " + i + "("
//  			   + tempI[i].numInstances());
//  	System.exit(0); 
//        } 
      

      // if we are using a smoothing metric, smooth the centroids
      if (m_metric instanceof SmoothingMetric &&
	  ((SmoothingMetric) m_metric).getUseSmoothing()) {
	System.out.println("\tSmoothing..."); 
	SmoothingMetric smoothingMetric = (SmoothingMetric) m_metric;
	centroid = smoothingMetric.smoothInstance(centroid); 
      }

      //   DEBUGGING:  replaced line under with block below
      m_ClusterCentroids.add(centroid); 
//        {
//  	tempNewCentroids.add(centroid);
//  	m_ClusterCentroids.delete(); 
//  	for (int j = 0; j <= i; j++) {
//  	  m_ClusterCentroids.add(tempNewCentroids.instance(j));
//  	}
//  	for (int j = i+1; j < m_NumClusters; j++) {
//  	  m_ClusterCentroids.add(tempCentroids.instance(j));
//  	} 
//  	double objBackup = m_Objective;
//  	System.out.println("  Updated cluster " + i + "("
//  			   + tempI[i].numInstances() + "); obj=" +
//  			   calculateObjectiveFunction(false));
//  	m_Objective = objBackup;
//        }
      
      // in SPKMeans, cluster centroids need to be normalized
      if (m_metric.doesNormalizeData()) {
	m_metric.normalizeInstanceWeighted(m_ClusterCentroids.instance(i));
      }
    }

    if (m_metric instanceof SmoothingMetric &&
	((SmoothingMetric) m_metric).getUseSmoothing())
      updateSmoothingMetrics();       
    
    for (int i = 0; i < m_NumClusters; i++)
      tempI[i] = null; // free memory
  }


  /** M-step of the KMeans clustering algorithm -- updates metric
   *  weights. Invoked only when we're using non-Potts model
   *  and metric is trainable
   */
  protected void updateMetricWeights() throws Exception {
    if (m_useMultipleMetrics) {
      for (int i = 0; i < m_NumClusters; i++) {
	m_metricLearners[i].trainMetric(i);
      } 
    } else {
      m_metricLearner.trainMetric(-1);
    } 
    InitNormalizerRegularizer();
  }
 

  /** checks for convergence */
  public boolean convergenceCheck(double oldObjective,
				  double newObjective) throws Exception {
    boolean converged = false;

    // Convergence check
    if(Math.abs(oldObjective - newObjective) < m_ObjFunConvergenceDifference) {
      System.out.println("Final objective function is: " + newObjective);
      converged = true;
    }

    // number of iterations check
    if (m_numBlankIterations >= m_maxBlankIterations) {
      System.out.println("Max blank iterations reached ...\n");
      System.out.println("Final objective function is: " + newObjective);
      converged = true;
    }
    if (m_Iterations >= m_maxIterations) {
      System.out.println("Max iterations reached ...\n");
      System.out.println("Final objective function is: " + newObjective);
      converged = true;
    }

    return converged;
  }

  /** calculates objective function */
  public double calculateObjectiveFunction(boolean isComplete) throws Exception {
    System.out.println("\tCalculating objective function ...");

    // update the oldObjective only if previous estimate of m_Objective
    // was complete
    if (isComplete) {
      m_OldObjective = m_Objective;
    }
    m_Objective = 0;
    m_objVariance = 0;
    m_objMustLinks = 0;
    m_objCannotLinks = 0;
    m_objNormalizer = 0;

    // Some debugging code:  tracking per-cluster objective
    double[] objectives = new double[m_NumClusters]; 

    // temporarily halve weights since every constraint is counted twice
    double tempML = m_MLweight;
    double tempCL = m_CLweight;
    m_MLweight = tempML/2;
    m_CLweight = tempCL/2; 
    
    if (m_verbose) {
      System.out.println("Must link weight: " + m_MLweight);
      System.out.println("Cannot link weight: " + m_CLweight);    
    }

    for (int i=0; i<m_Instances.numInstances(); i++) {
      if (m_isOfflineMetric) {
	double dist = m_metric.penalty(m_Instances.instance(i),
				       m_ClusterCentroids.instance(m_ClusterAssignments[i]));
	m_Objective += dist;
	if (m_verbose) {
	  System.out.println("Component for " + i + " = " + dist);
	}
      }
      else {
	double penalty = penaltyForInstance(i, m_ClusterAssignments[i]);
	objectives[m_ClusterAssignments[i]] += penalty;
	m_Objective += penalty; 
	m_objVariance += m_objVarianceCurrPoint;
	m_objMustLinks += m_objMustLinksCurrPoint;
	m_objCannotLinks += m_objCannotLinksCurrPoint;
	m_objNormalizer += m_objNormalizerCurrPoint;
      }
    }

    m_Objective -= m_objRegularizer;

    m_MLweight = tempML;
    m_CLweight = tempCL; // reset the values of the constraint weights

    // debugging:  reporting per-cluster objectives
    for (int i = 0; i < m_NumClusters; i++) {
      System.out.println("\t\tCluster " + i + " obj=" + objectives[i]); 
    }
    System.out.println("\tTotalObj=" + m_Objective); 

    // Oscillation check
    if ((float)m_OldObjective < (float)m_Objective) {
      System.out.println("WHOA!!!  Oscillations => bug in EM step?");
      System.out.println("Old objective:" + (float)m_OldObjective
			 + " < New objective: " + (float)m_Objective); 
    }

//      // TEMPORARY BLAH
//      System.out.println("\tvar=" + ((float)m_objVariance)
//  			 + "\tC=" + ((float)m_objCannotLinks)
//  			 + "\tM=" + ((float)m_objMustLinks)
//  			 + "\tLOG=" + ((float)m_objNormalizer) 
//  			 + "\tREG=" + ((float)m_objRegularizer));


    return m_Objective;
  }

  
  /** Actual KMeans function */
  protected void runKMeans() throws Exception {
    boolean converged = false;
    m_Iterations = 0;
    m_numBlankIterations = 0; 
    m_Objective = Double.POSITIVE_INFINITY; 

    if (!m_isOfflineMetric) {
      if (m_useMultipleMetrics) {
	for (int i = 0; i < m_metrics.length; i++) {
	  m_metrics[i].resetMetric();
	  m_metricLearners[i].resetLearner();
	} 
      } else { 
	m_metric.resetMetric();
	m_metricLearner.resetLearner();
      }
      // initialize max CL penalties
      if (m_ConstraintsHash.size() > 0) {
	m_maxCLPenalties = calculateMaxCLPenalties();
      }
    }

    // initialize m_ClusterAssignments
    for (int i=0; i<m_NumClusters; i++) {
      m_ClusterAssignments[i] = -1;
    }


    PrintStream fincoh = null;
    if (m_ConstraintIncoherenceFile != null) {
      fincoh = new PrintStream(new FileOutputStream(m_ConstraintIncoherenceFile));
    }

    while (!converged) {
      System.out.println("\n" + m_Iterations + ". Objective function: " + ((float)m_Objective));
      m_OldObjective = m_Objective; 
      
      // E-step
      int numMovedPoints = findBestAssignments();

      m_numBlankIterations = (numMovedPoints == 0) ? m_numBlankIterations+1 : 0; 

      //      calculateObjectiveFunction(false);
      System.out.println((float)m_Objective + " - Objective function after point assignment(CALC)");
      System.out.println("\tvar=" + ((float)m_objVariance) 
			 + "\tC=" + ((float)m_objCannotLinks) 
			 + "\tM=" + ((float)m_objMustLinks) 
			 + "\tLOG=" + ((float)m_objNormalizer) 
			 + "\tREG=" + ((float)m_objRegularizer));
     

      // M-step
      updateClusterCentroids();

      //      calculateObjectiveFunction(false);
      System.out.println((float)m_Objective + " - Objective function after centroid estimation");
      System.out.println("\tvar=" + ((float)m_objVariance)
			 + "\tC=" + ((float)m_objCannotLinks)
			 + "\tM=" + ((float)m_objMustLinks)
			 + "\tLOG=" + ((float)m_objNormalizer) 
			 + "\tREG=" + ((float)m_objRegularizer));
      
      if (m_Trainable == TRAINING_INTERNAL && !m_isOfflineMetric) {
	updateMetricWeights();
	if (m_verbose) {
	  calculateObjectiveFunction(true);
	  System.out.println((float)m_Objective + " - Objective function after metric update");
	  System.out.println("\tvar=" + ((float)m_objVariance) + "\tC=" + ((float)m_objCannotLinks) +
			     "\tM=" + ((float)m_objMustLinks)  + "\tLOG=" + ((float)m_objNormalizer) +
			     "\tREG=" + ((float)m_objRegularizer));
	}
	  
	if (m_ConstraintsHash.size() > 0) {
	  m_maxCLPenalties = calculateMaxCLPenalties();
	}
      }

      if (fincoh != null) {
	  printConstraintIncoherence(fincoh);
      }

      converged = convergenceCheck(m_OldObjective, m_Objective);
      m_Iterations++;
    }

    if (fincoh != null) {
      fincoh.close();
    }
    System.out.println("Converged!");
    System.err.print("Its\t" + m_Iterations + "\t");

    if (m_verbose) {
      System.out.println("Done clustering; top cluster features: ");
      for (int i = 0; i < m_NumClusters; i++){
	System.out.println("Centroid " + i);
	TreeMap map = new TreeMap(Collections.reverseOrder());
	Instance centroid= m_ClusterCentroids.instance(i);
	for (int j = 0; j < centroid.numValues(); j++) {
	  Attribute attr = centroid.attributeSparse(j);
	  map.put(new Double(centroid.value(attr)), attr.name());
	}
	Iterator it = map.entrySet().iterator();
	for (int j=0; j < 5 && it.hasNext(); j++) {
	  Map.Entry entry = (Map.Entry) it.next();
	  System.out.println("\t" + entry.getKey() + "\t" + entry.getValue());
	}
      }
    }
  }


  public void printConstraintIncoherence(PrintStream fincoh) throws Exception {
      Object[] array = m_ConstraintsHash.entrySet().toArray();
      
      int numML = 0, numCL = 0; 
      double incoh = 0;
      
      m_numViolations = 0;

      System.out.println("NumConstraints: " + array.length);
      for (int i=0; i < array.length; i++) {
	  Map.Entry con1 = (Map.Entry) array[i];
	  InstancePair pair1 = (InstancePair) con1.getKey();
	  int link1 = ((Integer) con1.getValue()).intValue();
	  double dist1 =  m_metric.distance(m_Instances.instance(pair1.first),
					    m_Instances.instance(pair1.second));
	  if (link1 == InstancePair.MUST_LINK) {
	      numML++;
	  } else if (link1 == InstancePair.CANNOT_LINK) {
	      numCL++;
	  }

	  for (int j=i+1; j < array.length; j++) {
	      Map.Entry con2 = (Map.Entry) array[j];
	      InstancePair pair2 = (InstancePair) con2.getKey();
	      int link2 = ((Integer) con2.getValue()).intValue();
	      double dist2 =  m_metric.distance(m_Instances.instance(pair2.first),
						m_Instances.instance(pair2.second));
	      
	      if (link1 == InstancePair.MUST_LINK) {
		  if (link2 == InstancePair.CANNOT_LINK) {
		      if (dist1 > dist2) {
			  m_numViolations++;
			  //			  System.out.println("(" + pair1.first + "," + pair1.second + "): " + link1 + ":" + dist1);
			  //			  System.out.println("(" + pair2.first + "," + pair2.second + "): " + link2 + ":" + dist2);
			  //			  System.out.println("Violations: " + m_numViolations);
		      }
		  }
	      } else if (link1 == InstancePair.CANNOT_LINK) {
		  if (link2 == InstancePair.MUST_LINK) {
		      if (dist1 < dist2) {
			  m_numViolations++;
			  //			  System.out.println("(" + pair1.first + "," + pair1.second + "): " + link1 + ":" + dist1);
			  //			  System.out.println("(" + pair2.first + "," + pair2.second + "): " + link2 + ":" + dist2);
			  //			  System.out.println("Violations: " + m_numViolations);
		      }
		  }
	      }
	  }
      }

      incoh = (m_numViolations * 1.0) / (numCL * numML);

      if (fincoh != null) {
	  //	  fincoh.println((m_Iterations+1)  + "\tNumViolations\t" + m_numViolations + "\tNumTotalCL\t" + numCL + "\tNumTotalML\t" + numML);
	  fincoh.println("Iterations\t" + (m_Iterations+1)  + "\tIncoh\t" + incoh);
      } else {
	  System.out.println((m_Iterations+1) + "\tNumViolations\t" + m_numViolations + "\tNumTotalCL\t" + numCL + "\tNumTotalML\t" + numML);
      }
  }


  /** reset the value of the objective function and all of its components */ 
  public void resetObjective() { 
    m_Objective = 0;
    m_objVariance = 0;
    m_objCannotLinks = 0;
    m_objMustLinks = 0;
    m_objNormalizer = 0;
    m_objRegularizer = 0;
  }
  
  /** Go through the cannot-link constraints and find the current maximum distance
   * @return an array of maximum weighted distances.  If a single metric is used, maximum distance
   * is calculated over the entire dataset */
  // TODO:  non-datasetWide case is not debugged currently!!!
  protected double[] calculateMaxCLPenalties() throws Exception {
    double [] maxPenalties = null;
    double [][] minValues = null;
    double [][] maxValues = null;
    int[] attrIdxs = null; 

    maxPenalties = new double[m_NumClusters];
    m_maxCLPoints = new Instance[m_NumClusters][2];
    m_maxCLDiffInstances = new Instance[m_NumClusters];

    for (int i = 0; i < m_NumClusters; i++) { 
      m_maxCLPoints[i][0] = new Instance(m_Instances.numAttributes());
      m_maxCLPoints[i][1] = new Instance(m_Instances.numAttributes());
      m_maxCLPoints[i][0].setDataset(m_Instances);
      m_maxCLPoints[i][1].setDataset(m_Instances);
      m_maxCLDiffInstances[i] = new Instance(m_Instances.numAttributes());
      m_maxCLDiffInstances[i].setDataset(m_Instances);
    }

    // TEMPORARY PLUG:  this was supposed to take care of WeightedDotp,
    // but it turns out that with weighting similarity can be > 1. 
//      if (m_metric.m_fixedMaxDistance) {
//        for (int i = 0; i < m_NumClusters; i++) {
//  	maxPenalties[i] = m_metric.getMaxDistance(); 
//        }
//        return maxPenalties; 
//      } 

    
    minValues = new double[m_NumClusters][m_metrics[0].getNumAttributes()];
    maxValues = new double[m_NumClusters][m_metrics[0].getNumAttributes()];
    attrIdxs = m_metrics[0].getAttrIndxs();

    // temporary plug:  if this if the first iteration when no instances were assigned to clusters,
    // dataset-wide (not cluster-wide!) minimum and maximum are used even for the case with
    // multiple metrics
    boolean datasetWide = true;
    if (m_useMultipleMetrics && m_Iterations > 0) { 
      datasetWide = false;
    } 

    // TODO:  Mahalanobis - check with getMaxPoints
    // go through all points
    if (m_metric instanceof WeightedMahalanobis) {
      if (m_useMultipleMetrics) { 
	for (int i = 0; i < m_metrics.length; i++) { 
	  double[][] maxPoints = ((WeightedMahalanobis)m_metrics[i]).getMaxPoints(m_ConstraintsHash, m_Instances);
	  minValues[i] = maxPoints[0];
	  maxValues[i] = maxPoints[1];
	  //  	  System.out.println("Max points " + i);
	  //  	  for (int j = 0; j < maxPoints[0].length; j++) { System.out.println(maxPoints[0][j] + " - " + maxPoints[1][j]);}
	}
      } else { 
	double[][] maxPoints = ((WeightedMahalanobis)m_metric).getMaxPoints(m_ConstraintsHash, m_Instances);
	minValues[0] = maxPoints[0];
	maxValues[0] = maxPoints[1];
	for (int i = 0; i < m_metrics.length; i++) {
	  minValues[i] = maxPoints[0];
	  maxValues[i] = maxPoints[1];
	}
	//  	System.out.println("Max points:");
	//  	for (int i = 0; i < maxPoints[0].length; i++) { System.out.println(maxPoints[0][i] + " - " + maxPoints[1][i]);}
      }
    } else { // find the enclosing hypercube for WeightedEuclidean etc. 
      for (int i = 0; i < m_Instances.numInstances(); i++) {
	Instance instance = m_Instances.instance(i);
	for (int j = 0; j < attrIdxs.length; j++) {
	  double val = instance.value(attrIdxs[j]);
	  if (datasetWide) {
	    if (val < minValues[0][j]) {
	      minValues[0][j] = val; 
	    }
	    if (val > maxValues[0][j]) {
	      maxValues[0][j] = val; 
	    } 
	  } else { // cluster-specific min's and max's  are needed
	    if (val < minValues[m_ClusterAssignments[i]][j]) {
	      minValues[m_ClusterAssignments[i]][j] = val; 
	    }
	    if (val > maxValues[m_ClusterAssignments[i]][j]) {
	      maxValues[m_ClusterAssignments[i]][j] = val; 
	    } 
	  } 
	}
      }
    }

    // get the max/min points
    if (datasetWide) {
      for (int i = 0; i < attrIdxs.length; i++) {
	m_maxCLPoints[0][0].setValue(attrIdxs[i], minValues[0][i]);
	m_maxCLPoints[0][1].setValue(attrIdxs[i], maxValues[0][i]);
      }
      // must copy these over all clusters - just for the first iteration
      for (int j = 1; j < m_NumClusters; j++) { 
	for (int i = 0; i < attrIdxs.length; i++) {
	  m_maxCLPoints[j][0].setValue(attrIdxs[i], minValues[0][i]);
	  m_maxCLPoints[j][1].setValue(attrIdxs[i], maxValues[0][i]);
	}
      } 
    } else { // cluster-specific
      for (int j = 0; j < m_NumClusters; j++) { 
	for (int i = 0; i < attrIdxs.length; i++) {
	  m_maxCLPoints[j][0].setValue(attrIdxs[i], minValues[j][i]);
	  m_maxCLPoints[j][1].setValue(attrIdxs[i], maxValues[j][i]);
	}
      }
    }

    // calculate the distances
    if (datasetWide) {
      maxPenalties[0] = m_metrics[0].penaltySymmetric(m_maxCLPoints[0][0],
						      m_maxCLPoints[0][1]);
      m_maxCLDiffInstances[0] = m_metrics[0].createDiffInstance(m_maxCLPoints[0][0], 
								 m_maxCLPoints[0][1]);
      for (int i = 1; i < maxPenalties.length; i++) {
	maxPenalties[i] = maxPenalties[0];
	m_maxCLDiffInstances[i] = m_maxCLDiffInstances[0];
      } 
    } else { // cluster-specific - SHOULD BE FIXED!!!!
      for (int j = 0; j < m_NumClusters; j++) { 
	for (int i = 0; i < attrIdxs.length; i++) {
	  maxPenalties[j] += m_metrics[j].penaltySymmetric(m_maxCLPoints[j][0],
							   m_maxCLPoints[j][1]);
	  m_maxCLDiffInstances[j] = m_metrics[0].createDiffInstance(m_maxCLPoints[j][0], 
								    m_maxCLPoints[j][1]);
	}
      }
    }
    System.out.println("Recomputed max CL penalties");
    return maxPenalties;
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
    return assignInstanceToCluster(instance);
  }

  /** lookup the instance in the checksum hash, assuming transductive clustering
   * @param instance instance to be looked up
   * @return the index of the cluster to which the instance was assigned, -1 if the instance has not bee clustered
   */
  protected int lookupInstanceCluster(Instance instance) throws Exception {
    int classIdx = instance.classIndex();
    double checksum = 0;

    // need to normalize using original metric, since cluster data is normalized similarly
    if (m_metric.doesNormalizeData()) {
      if (m_Trainable == TRAINING_INTERNAL) {
	m_metric.resetMetric();
      }
      m_metric.normalizeInstanceWeighted(instance);
    }

    double[] values1 = instance.toDoubleArray();
    for (int i = 0; i < values1.length; i++) {
      if (i != classIdx) {
	checksum += m_checksumCoeffs[i] * values1[i]; 
      } 
    }

    Object list = m_checksumHash.get(new Double((float)checksum));
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
	    if ((float)values1[j] != (float)values2[j]) {
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
   * Classifies the instances using the current clustering, moves
   * must-linked points together (Xing's approach)
   *
   * @param instIdx the instance index to be assigned to a cluster
   * @return the number of the assigned cluster as an integer
   * if the class is enumerated, otherwise the predicted value
   * @exception Exception if instance could not be classified
   * successfully */

  public int assignAllInstancesToClusters() throws Exception {
    int numInstances = m_Instances.numInstances();
    boolean [] instanceAlreadyAssigned = new boolean[numInstances];
    int moved = 0;

    if (!m_isOfflineMetric) {
      System.err.println("WARNING!!!\n\nThis code should not be called if metric is not a BarHillelMetric or XingMetric!!!!\n\n");
    }

    for (int i=0; i<numInstances; i++) {
      instanceAlreadyAssigned[i] = false;
    }

    // now process points not in ML meighborhood sets
    for (int instIdx = 0; instIdx < numInstances; instIdx++) {
      if (instanceAlreadyAssigned[instIdx]) { 
	continue; // was already in some ML neighborhood
      }
      int bestCluster = 0;
      double bestDistance = Double.POSITIVE_INFINITY;
      for (int centroidIdx = 0; centroidIdx < m_NumClusters; centroidIdx++) {
	double sqDistance = m_metric.distance(m_Instances.instance(instIdx), m_ClusterCentroids.instance(centroidIdx));
	if (sqDistance < bestDistance) {
	  bestDistance = sqDistance;
	  bestCluster = centroidIdx;
	}
      }

      // accumulate objective function value
      //      m_Objective += bestDistance;

      // do we need to reassign the point?
      if (m_ClusterAssignments[instIdx] != bestCluster) {
	m_ClusterAssignments[instIdx] = bestCluster;
	instanceAlreadyAssigned[instIdx] = true;
	moved++;
      }
    }
    return moved;
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
    int lookupCluster;

    if (m_metric instanceof InstanceConverter) {
      Instance newInstance = ((InstanceConverter)m_metric).convertInstance(instance);
      lookupCluster = lookupInstanceCluster(newInstance);
    } else {
      lookupCluster = lookupInstanceCluster(instance);
    }
    if (lookupCluster >= 0) {
      return lookupCluster;
    }
    throw new Exception ("ACHTUNG!!!\n\nCouldn't lookup the instance!!! Size of hash = " + m_checksumHash.size());
  }
  
  /** Set the cannot link constraint weight */
  public void setCannotLinkWeight(double w) {
    m_CLweight = w;
  }

  /** Return the cannot link constraint weight */
  public double getCannotLinkWeight() {
    return m_CLweight;
  }

  /** Set the must link constraint weight */
  public void setMustLinkWeight(double w) {
    m_MLweight = w;
  }

  /** Return the must link constraint weight */
  public double getMustLinkWeight() {
    return m_MLweight;
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

  /** Set the maximum number of iterations */
  public void setMaxIterations(int maxIterations) {
    m_maxIterations = maxIterations;
  }

  /** Get the maximum number of iterations */
  public int getMaxIterations() {
    return m_maxIterations;
  }

  /** Set the maximum number of blank iterations (those where no points are moved) */
  public void setMaxBlankIterations(int maxBlankIterations) {
    m_maxBlankIterations = maxBlankIterations;
  }

  /** Get the maximum number of blank iterations */
  public int getMaxBlankIterations() {
    return m_maxBlankIterations;
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
      Object list = m_checksumHash.get(new Double((float)checksum));
      ArrayList idxList = null; 
      if (list == null) {
	idxList = new ArrayList();
	m_checksumHash.put(new Double((float)checksum), idxList);
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

  /** Is the objective function decreasing or increasing? */
  public boolean isObjFunDecreasing() {
    return m_objFunDecreasing;
  } 


  /**
   * Set the distance metric
   *
   * @param s the metric
   */
  public void setMetric (LearnableMetric m) {
    String metricName = m.getClass().getName();
    m_metric = m;
    m_metricLearner.setMetric(m_metric);
    m_metricLearner.setClusterer(this);
  }

  /**
   * get the distance metric
   * @returns the distance metric used
   */
  public LearnableMetric getMetric () {
    return m_metric;
  }

  /**
   * get the array of metrics
   */
  public LearnableMetric[] getMetrics () {
    return m_metrics;
  }

  /** Set/get the metric learner */
  public void setMetricLearner (MPCKMeansMetricLearner ml) {
    m_metricLearner = ml;
    m_metricLearner.setMetric(m_metric);
    m_metricLearner.setClusterer(this);
  }
  public MPCKMeansMetricLearner getMetricLearner () {
    return m_metricLearner;
  }

  /** Set/get the assigner */
  public MPCKMeansAssigner getAssigner() {
    return m_Assigner;
  }
  public void setAssigner(MPCKMeansAssigner assigner) {
    assigner.setClusterer(this);
    this.m_Assigner = assigner;
  }

  /** Set/get the initializer */
  public MPCKMeansInitializer getInitializer() {
    return m_Initializer;
  }

  public void setInitializer(MPCKMeansInitializer initializer) {
    initializer.setClusterer(this);
    this.m_Initializer = initializer;
  }

  /** Read the seeds from a hastable, where every key is an instance and every value is:
   * the cluster assignment of that instance 
   * seedVector vector containing seeds
   */
  
  public void seedClusterer(HashMap seedHash) {
    System.err.println("Not implemented here");
  }
 
  public void printClusterAssignments() throws Exception {
    if (m_ClusterAssignmentsOutputFile != null) {
      PrintStream p = 
	new PrintStream(new FileOutputStream(m_ClusterAssignmentsOutputFile));
      
      for (int i=0; i<m_Instances.numInstances(); i++) {
	p.println(i + "\t" + m_ClusterAssignments[i]);
      }
      p.close();
    } else {
      System.out.println("\nCluster Assignments:\n");
      for (int i=0; i<m_Instances.numInstances(); i++) {
	System.out.println(i + "\t" + m_ClusterAssignments[i]);
      }
    }
  }


  /** Prints clusters */
  public void printClusters () throws Exception {
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
      if (m_verbose) {
	//	System.out.println("In getIndexClusters, " + i + " assigned to cluster " + m_ClusterAssignments[i]);
      }
      if (m_ClusterAssignments[i]!=-1 && m_ClusterAssignments[i] < m_NumClusters) {
	if (m_IndexClusters[m_ClusterAssignments[i]] == null) {
	  m_IndexClusters[m_ClusterAssignments[i]] = new HashSet();
	}
	m_IndexClusters[m_ClusterAssignments[i]].add(new Integer(i));
      }
    }
    return m_IndexClusters;
  }


  public Enumeration listOptions () {
    return null;
  }

  public String [] getOptions () {
    String[] options = new String[150];
    int current = 0;

    if (!m_Seedable) {
      options[current++] = "-X";
    }

    if (m_Trainable != TRAINING_NONE) {
      options[current++] = "-T";
      if (m_Trainable == TRAINING_INTERNAL) { 
	options[current++] = "Int";
      } else {
	options[current++] = "Ext";
      }
    }

    options[current++] = "-M";
    options[current++] = Utils.removeSubstring(m_metric.getClass().getName(), "weka.core.metrics.");
    if (m_metric instanceof OptionHandler) {
      String[] metricOptions = ((OptionHandler)m_metric).getOptions();
      for (int i = 0; i < metricOptions.length; i++) {
	options[current++] = metricOptions[i];
      }
    } 

    if (m_Trainable != TRAINING_NONE) {
      options[current++] = "-L";
      options[current++] = Utils.removeSubstring(m_metricLearner.getClass().getName(), "weka.clusterers.metriclearners.");
      String[] metricLearnerOptions = ((OptionHandler)m_metricLearner).getOptions();
      for (int i = 0; i < metricLearnerOptions.length; i++) {
	options[current++] = metricLearnerOptions[i];
      }
    }

    if (m_regularize) {
      options[current++] = "-G";
      options[current++] = Utils.removeSubstring(m_metric.getRegularizer().getClass().getName(), "weka.clusterers.regularizers.");
      if (m_metric.getRegularizer() instanceof OptionHandler) { 
	String[] regularizerOptions = ((OptionHandler)m_metric.getRegularizer()).getOptions();
	for (int i = 0; i < regularizerOptions.length; i++) {
	  options[current++] = regularizerOptions[i];
	}
      }
    } 

    options[current++] = "-A";
    options[current++] = Utils.removeSubstring(m_Assigner.getClass().getName(), "weka.clusterers.assigners.");
    if (m_Assigner instanceof OptionHandler) {
      String[] assignerOptions = ((OptionHandler)m_Assigner).getOptions();
      for (int i = 0; i < assignerOptions.length; i++) {
	options[current++] = assignerOptions[i];
      }
    }

    options[current++] = "-I";
    options[current++] = Utils.removeSubstring(m_Initializer.getClass().getName(), "weka.clusterers.initializers.");
    if (m_Initializer instanceof OptionHandler) {
      String[] initializerOptions = ((OptionHandler)m_Initializer).getOptions();
      for (int i = 0; i < initializerOptions.length; i++) {
	options[current++] = initializerOptions[i];
      }
    }

    if (m_useMultipleMetrics) {
      options[current++] = "-U";
    }

    options[current++] = "-N";
    options[current++] = "" + getNumClusters();
    options[current++] = "-R";
    options[current++] = "" + getRandomSeed();

    options[current++] = "-l";
    options[current++] = "" + m_logTermWeight;
    options[current++] = "-r";
    options[current++] = "" + m_regularizerTermWeight;
    options[current++] = "-m";
    options[current++] = "" + m_MLweight;
    options[current++] = "-c";
    options[current++] = "" + m_CLweight;

    options[current++] = "-i";
    options[current++] = "" + m_maxIterations;
    options[current++] = "-B";
    options[current++] = "" + m_maxBlankIterations;

    options[current++] = "-O";
    options[current++] = "" + m_ClusterAssignmentsOutputFile;
    options[current++] = "-H";
    options[current++] = "" + m_ConstraintIncoherenceFile;
    options[current++] = "-V";
    options[current++] = "" + m_useTransitiveConstraints;

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
  public void setOptions (String[] options) throws Exception {
    if (Utils.getFlag('X', options)) {
      System.out.println("Setting seedable to: false");
      setSeedable(false);
    }

    String optionString = Utils.getOption('T', options);
    if (optionString.length() != 0) {
      setTrainable(new SelectedTag(Integer.parseInt(optionString), TAGS_TRAINING));
      System.out.println("Setting trainable to: " + Integer.parseInt(optionString));
    }

    optionString = Utils.getOption('M', options);
    if (optionString.length() != 0) {
      String[] metricSpec = Utils.splitOptions(optionString);
      String metricName = metricSpec[0]; 
      metricSpec[0] = "";
      setMetric((LearnableMetric) Utils.forName(LearnableMetric.class, 
						metricName, metricSpec));
      System.out.println("Setting metric to: " + metricName);
    }

    optionString = Utils.getOption('L', options);
    if (optionString.length() != 0) {
      String[] learnerSpec = Utils.splitOptions(optionString);
      String learnerName = learnerSpec[0]; 
      learnerSpec[0] = "";
      setMetricLearner((MPCKMeansMetricLearner) Utils.forName(MPCKMeansMetricLearner.class,
							      learnerName, learnerSpec));
      System.out.println("Setting metricLearner to: " + m_metricLearner);
    }

    optionString = Utils.getOption('G', options);
    if (optionString.length() != 0) {
      String[] regularizerSpec = Utils.splitOptions(optionString);
      String regularizerName = regularizerSpec[0]; 
      regularizerSpec[0] = "";
      m_metric.setRegularizer((Regularizer) Utils.forName(Regularizer.class,
							  regularizerName, regularizerSpec));
      System.out.println("Setting regularizer to: " + regularizerName);
    }

    optionString = Utils.getOption('A', options);
    if (optionString.length() != 0) {
      String[] assignerSpec = Utils.splitOptions(optionString);
      String assignerName = assignerSpec[0]; 
      assignerSpec[0] = "";
      setAssigner((MPCKMeansAssigner) Utils.forName(MPCKMeansAssigner.class,
						    assignerName, assignerSpec));
      System.out.println("Setting assigner to: " + assignerName);
    }
    
    optionString = Utils.getOption('I', options);
    if (optionString.length() != 0) {
      String[] initializerSpec = Utils.splitOptions(optionString);
      String initializerName = initializerSpec[0]; 
      initializerSpec[0] = "";
      setInitializer((MPCKMeansInitializer) Utils.forName(MPCKMeansInitializer.class,
							  initializerName, initializerSpec));
      System.out.println("Setting initializer to: " + initializerName);
    }

    if (Utils.getFlag('U', options)) {
      setUseMultipleMetrics(true);
      System.out.println("Setting multiple metrics to: true");
    }

    optionString = Utils.getOption('N', options);
    if (optionString.length() != 0) {
      setNumClusters(Integer.parseInt(optionString));
      System.out.println("Setting numClusters to: " + m_NumClusters);
    } 

    optionString = Utils.getOption('R', options);
    if (optionString.length() != 0) {
      setRandomSeed(Integer.parseInt(optionString));
      System.out.println("Setting randomSeed to: " + m_RandomSeed);
    }

    optionString = Utils.getOption('l', options);
    if (optionString.length() != 0) {
      setLogTermWeight(Double.parseDouble(optionString));
      System.out.println("Setting logTermWeight to: " + m_logTermWeight);
    }

    optionString = Utils.getOption('r', options);
    if (optionString.length() != 0) {
      setRegularizerTermWeight(Double.parseDouble(optionString));
      System.out.println("Setting regularizerTermWeight to: " 
			 + m_regularizerTermWeight);
    }

    optionString = Utils.getOption('m', options);
    if (optionString.length() != 0) {
      setMustLinkWeight(Double.parseDouble(optionString));
      System.out.println("Setting mustLinkWeight to: " + m_MLweight);
    }

    optionString = Utils.getOption('c', options);
    if (optionString.length() != 0) {
      setCannotLinkWeight(Double.parseDouble(optionString));
      System.out.println("Setting cannotLinkWeight to: " + m_CLweight);
    }

    optionString = Utils.getOption('i', options);
    if (optionString.length() != 0) {
      setMaxIterations(Integer.parseInt(optionString));
      System.out.println("Setting maxIterations to: " + m_maxIterations);
    }

    optionString = Utils.getOption('B', options);
    if (optionString.length() != 0) {
      setMaxBlankIterations(Integer.parseInt(optionString));
      System.out.println("Setting maxBlankIterations to: " + m_maxBlankIterations);
    }

    optionString = Utils.getOption('O', options);
    if (optionString.length() != 0) {
      setClusterAssignmentsOutputFile(optionString);
      System.out.println("Setting clusterAssignmentsOutputFile to: " 
			 + m_ClusterAssignmentsOutputFile);
    }

    optionString = Utils.getOption('H', options);
    if (optionString.length() != 0) {
      setConstraintIncoherenceFile(optionString);
      System.out.println("Setting m_ConstraintIncoherenceFile to: " 
			 + m_ConstraintIncoherenceFile);
    }


    if (Utils.getFlag('V', options)) {
      setUseTransitiveConstraints(false);
      System.out.println("Setting useTransitiveConstraints to: false");
    }
  }
  
  /**   
   * return a string describing this clusterer
   *
   * @return a description of the clusterer as a string
   */
  public String toString() {
    StringBuffer temp = new StringBuffer();
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

  /** Set/get the use of transitive closure */
  public void setUseTransitiveConstraints(boolean useTransitiveConstraints) {
    m_useTransitiveConstraints = useTransitiveConstraints;
  }

  public boolean getUseTransitiveConstraints() {
    return m_useTransitiveConstraints;
  } 

  /**
   * Turn on/off the use of per-cluster metrics
   * @param useMultipleMetrics if true, individual metrics will be used for each cluster
   */
  public void setUseMultipleMetrics (boolean useMultipleMetrics) {
    m_useMultipleMetrics = useMultipleMetrics;
  }

  /**
   * See if individual per-cluster metrics are used
   * @return true if individual metrics are used for each cluster
   */
  public boolean getUseMultipleMetrics () {
    return m_useMultipleMetrics;
  }


  /**
   * Turn on/off the use of regularization of weights
   * @param regularize, if true weights will be regularized
   */
  public void setRegularize (boolean regularize) {
    m_regularize = regularize;
  }

  /**
   * See if weights are regularized
   * @return true if weights are regularized
   */
  public boolean getRegularize () {
    return m_regularize;
  }


  /**
   * Get the value of the weight assigned to log term in the objective function
   * @return value of the weight assigned to log term in the objective function
   */
  public double getLogTermWeight() {
    return m_logTermWeight;
  }
  
  /**
   * Set the value of the weight assigned to log term in the objective function
   * @param logTermWeight weight assigned to log term in the objective function
   */
  public void setLogTermWeight(double  logTermWeight) {
    this.m_logTermWeight = logTermWeight;
  }

  /**
   * Get the value of the weight assigned to regularizer term in the objective function
   * @return value of the weight assigned to regularizer term in the objective function
   */
  public double getRegularizerTermWeight() {
    return m_regularizerTermWeight;
  }
  
  /**
   * Set the value of the weight assigned to regularizer term in the objective function
   * @param regularizerTermWeight weight assigned to regularizer term in the objective function
   */
  public void setRegularizerTermWeight(double  regularizerTermWeight) {
    this.m_regularizerTermWeight = regularizerTermWeight;
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

  /** Read constraints from a file */
  public ArrayList readConstraints(String fileName) {
    ArrayList pairs = new ArrayList();

    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String s = null;
      int first = 0, second = 0, constraint = InstancePair.DONT_CARE_LINK;
      InstancePair pair = null;

      while ((s = reader.readLine()) != null) {
	StringTokenizer tokenizer = new StringTokenizer(s);
	int i = 0;
	while (tokenizer.hasMoreTokens()) {
	  String token = tokenizer.nextToken();
	  if (i == 0) {
	    first = Integer.parseInt(token);
	    //	    System.out.println("First instance: " + first);
	  } else if (i == 1) {
	    second = Integer.parseInt(token);
	    //	    System.out.println("Second instance: " + second);
	  } else if (i == 2) {
	    constraint = Integer.parseInt(token);
	    if (constraint < 0) {
	      if (first < second) {
		pair = new InstancePair(first, second, InstancePair.CANNOT_LINK);
	      } else {
		pair = new InstancePair(second, first, InstancePair.CANNOT_LINK);
	      }
	      //	      System.out.println("CANNOT_LINK");
	    } else {
	      if (first < second) {
		pair = new InstancePair(first, second, InstancePair.MUST_LINK);
	      } else {
		pair = new InstancePair(second, first, InstancePair.MUST_LINK);
	      }
	      //	      System.out.println("MUST_LINK");
	    }
	    if (!pairs.contains(pair)) {
	      pairs.add(pair);
	    }
	  }
	  i++;
	}
      }
    } catch (Exception e) {
      System.out.println("Problems reading from constraints file: " + e);
      e.printStackTrace();
    }

    return pairs;
  }

  /**
   * Main method for testing this class.
   *
   */

  public static void main (String[] args) {
    //testCase();
    runFromCommandLine(args);
  }

  public static void runFromCommandLine(String[] args) {
    MPCKMeans mpckmeans = new MPCKMeans();
    Instances data = null, clusterData = null;
    ArrayList labeledPairs = null;

    try {
      String optionString = Utils.getOption('D', args);
      if (optionString.length() != 0) {
	FileReader reader = new FileReader (optionString);
	data = new Instances (reader);
	System.out.println("Reading dataset: " + data.relationName());
      }

      int classIndex = data.numAttributes()-1;
      optionString = Utils.getOption('K', args);
      if (optionString.length() != 0) {
	classIndex = Integer.parseInt(optionString);
	if (classIndex >= 0) { 
	  data.setClassIndex(classIndex); // starts with 0
	  // Remove the class labels before clustering
	  clusterData = new Instances(data);
	  mpckmeans.setNumClusters(clusterData.numClasses());
	  clusterData.deleteClassAttribute();
	  System.out.println("Setting classIndex: " + classIndex);
	} else {
	  clusterData = new Instances(data);
	}
      } else {
	data.setClassIndex(classIndex); // starts with 0
	// Remove the class labels before clustering
	clusterData = new Instances(data);
	mpckmeans.setNumClusters(clusterData.numClasses());
	clusterData.deleteClassAttribute();
	System.out.println("Setting classIndex: " + classIndex);
      }
      
      optionString = Utils.getOption('C', args);
      if (optionString.length() != 0) {
	labeledPairs = mpckmeans.readConstraints(optionString);
	System.out.println("Reading constraints from: " + optionString);
      } else {
	labeledPairs = new ArrayList(0);
      }

      mpckmeans.setTotalTrainWithLabels(data);
      mpckmeans.setOptions(args);
      System.out.println();
      mpckmeans.buildClusterer(labeledPairs, clusterData, data, mpckmeans.getNumClusters(), data.numInstances());
      mpckmeans.printClusterAssignments();

            if(mpckmeans.m_TotalTrainWithLabels.classIndex()>-1){
      double nCorrect = 0;
      for (int i=0; i<mpckmeans.m_TotalTrainWithLabels.numInstances(); i++) {
	  for (int j=i+1; j<mpckmeans.m_TotalTrainWithLabels.numInstances(); j++) {
	      int cluster_i = mpckmeans.m_ClusterAssignments[i];
	      int cluster_j = mpckmeans.m_ClusterAssignments[j];
	      double class_i = (mpckmeans.m_TotalTrainWithLabels.instance(i)).classValue();
	      double class_j = (mpckmeans.m_TotalTrainWithLabels.instance(j)).classValue();
	      //	      System.out.println(cluster_i + "," + cluster_j + ":" + class_i + "," + class_j);
	      if (cluster_i == cluster_j && class_i == class_j ||
		  cluster_i != cluster_j && class_i != class_j) {
		  nCorrect++;
		  //		  System.out.println("nCorrect:" + nCorrect);
	      }
	  }
      }
      int numInstances = mpckmeans.m_TotalTrainWithLabels.numInstances();
      double RandIndex = 100 * nCorrect/(numInstances*(numInstances-1)/2);
      System.err.println("Acc\t" + RandIndex);
            }

      //      if (mpckmeans.getTotalTrainWithLabels().classIndex() >= 0) {
      // 	SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(mpckmeans.m_TotalTrainWithLabels,
      // 									 mpckmeans.m_TotalTrainWithLabels.numClasses(),
      // 									 mpckmeans.m_TotalTrainWithLabels.numClasses());
      // 	eval.evaluateModel(mpckmeans, mpckmeans.m_TotalTrainWithLabels, mpckmeans.m_Instances);
      // 	eval.mutualInformation();
      // 	eval.pairwiseFMeasure();
      //      }
    } catch (Exception e) {
      System.out.println("Option not specified");
      e.printStackTrace();
    }
  }

  public static void testCase() {
    try {    
      String dataset = new String("lowd");
      //String dataset = new String("highd");
      if (dataset.equals("lowd")) {
	//////// Low-D data

	//	String datafile = "/u/ml/data/bio/arffFromPhylo/ecoli_K12-100.arff";
	//	String datafile = "/u/sugato/weka/data/digits-0.1-389.arff";
	String datafile = "/u/sugato/weka/data/iris.arff";
	int numPairs = 200, num=0;

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
		
	// create the pairs
	ArrayList labeledPair = InstancePair.getPairs(data,numPairs);

	System.out.println("Finished initializing constraint matrix");

	MPCKMeans mpckmeans = new MPCKMeans();
	mpckmeans.setUseMultipleMetrics(false);
	System.out.println("\nClustering the data using MPCKmeans...\n");

	WeightedEuclidean metric = new WeightedEuclidean();
	WEuclideanLearner metricLearner = new WEuclideanLearner();

	//  	LearnableMetric metric = new WeightedDotP();
	//  	MPCKMeansMetricLearner metricLearner = new DotPGDLearner();
	
	//  	KL metric = new KL();
	//  	KLGDLearner metricLearner = new KLGDLearner();
	//	((KL)metric).setUseIDivergence(true);

	//	BarHillelMetric metric = new BarHillelMetric();
	//	BarHillelMetricMatlab metric = new BarHillelMetricMatlab();
	//  	XingMetric metric = new XingMetric();
	//	WeightedMahalanobis metric = new WeightedMahalanobis(); 
	
	mpckmeans.setMetric(metric);
	mpckmeans.setMetricLearner(metricLearner);
	mpckmeans.setVerbose(false);
	mpckmeans.setRegularize(false);
	mpckmeans.setTrainable(new SelectedTag(TRAINING_INTERNAL, TAGS_TRAINING));
	mpckmeans.setSeedable(true);
	mpckmeans.buildClusterer(labeledPair, clusterData, data, data.numClasses(), data.numInstances());
	mpckmeans.getIndexClusters();
	mpckmeans.printIndexClusters();

	SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(mpckmeans.m_TotalTrainWithLabels,
									 mpckmeans.m_TotalTrainWithLabels.numClasses(),
									 mpckmeans.m_TotalTrainWithLabels.numClasses());
	eval.evaluateModel(mpckmeans, mpckmeans.m_TotalTrainWithLabels, mpckmeans.m_Instances);
	System.out.println("MI=" + eval.mutualInformation());
	System.out.print("FM=" + eval.pairwiseFMeasure());
	System.out.print("\tP=" + eval.pairwisePrecision());
	System.out.print("\tR=" + eval.pairwiseRecall());
      }
      else if (dataset.equals("highd")) {
	//////// Newsgroup data
	String datafile = "/u/ml/users/sugato/groupcode/weka335/data/arffFromCCS/sanitized/different-1000_sanitized.arff";
	//String datafile = "/u/ml/users/sugato/groupcode/weka335/data/20newsgroups/small-newsgroup_fromCCS.arff";
	//String datafile = "/u/ml/users/sugato/groupcode/weka335/data/20newsgroups/same-100_fromCCS.arff";
	
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
	
	// create the pairs
	int numPairs = 0, num=0;
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
	    //System.out.println(num + "th entry is: " + pair);
	    num++;
	  }
	}
	System.out.println("Finished initializing constraint matrix");
	
	MPCKMeans mpckmeans = new MPCKMeans();
	mpckmeans.setUseMultipleMetrics(false);
	System.out.println("\nClustering the highd data using MPCKmeans...\n");

  	LearnableMetric metric = new WeightedDotP();
  	MPCKMeansMetricLearner metricLearner = new DotPGDLearner();

//  	KL metric = new KL();
//  	KLGDLearner metricLearner = new KLGDLearner();

	mpckmeans.setMetric(metric);
	mpckmeans.setMetricLearner(metricLearner);
	mpckmeans.setVerbose(false);
	mpckmeans.setRegularize(true);
	mpckmeans.setTrainable(new SelectedTag(TRAINING_INTERNAL, TAGS_TRAINING));
	mpckmeans.setSeedable(true);
	mpckmeans.buildClusterer(labeledPair, clusterData, data, data.numClasses(), data.numInstances());
	mpckmeans.getIndexClusters();

	SemiSupClustererEvaluation eval = new SemiSupClustererEvaluation(mpckmeans.m_TotalTrainWithLabels,
									 mpckmeans.m_TotalTrainWithLabels.numClasses(),
									 mpckmeans.m_TotalTrainWithLabels.numClasses());
	mpckmeans.getMetric().resetMetric(); // Vital: to reset m_attrWeights to 1 for proper normalization
	eval.evaluateModel(mpckmeans, mpckmeans.m_TotalTrainWithLabels, mpckmeans.m_Instances);
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
