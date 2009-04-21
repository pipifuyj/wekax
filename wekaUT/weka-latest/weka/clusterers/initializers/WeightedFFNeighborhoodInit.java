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

/**
 *    WeightedFFNeighborhoodInit.java
 *
 *    Initializer that uses weighted farthest first traversal to get
 *    initial clusters for K-Means
 *
 *    Copyright (C) 2004 Sugato Basu, Misha Bilenko
 * */

package weka.clusterers.initializers; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.core.metrics.*;
import  weka.clusterers.*;

public class WeightedFFNeighborhoodInit extends MPCKMeansInitializer {
  /** holds the ([instance pair] -> [type of constraint]) mapping,
      where the hashed value stores the type of link but the instance
      pair does not hold the type of constraint - it holds (instanceIdx1,
      instanceIdx2, DONT_CARE_LINK). This is done to make lookup easier
      in future 
  */
  protected HashMap m_ConstraintsHash;

  /** stores the ([instanceIdx] -> [ArrayList of constraints])
      mapping, where the arraylist contains the constraints in which
      instanceIdx is involved. Note that the instance pairs stored in
      the Arraylist have the actual link type.  
  */
  protected HashMap m_instanceConstraintHash; 

  /** holds the points involved in the constraints */
  protected HashSet m_SeedHash;

  /** distance Metric */
  protected LearnableMetric m_metric;

  /** Is the objective function increasing or decreasing?  Depends on type
   * of metric used:  for similarity-based metric, increasing, for distance-based - decreasing */
  protected boolean m_objFunDecreasing;


  /** Seedable or not (true by default) */
  protected boolean m_Seedable = true;

  /** Number of clusters in the process*/
  protected int m_NumCurrentClusters = 0; 

  /**
   * holds the random number generator used in various parts of the code
   */
  protected Random m_RandomNumberGenerator;

  /** temporary variable holding cluster assignments while iterating */
  protected int [] m_ClusterAssignments;

  /** array holding sum of cluster instances */
  Instance [] m_SumOfClusterInstances;

  /** Instances without labels */
  protected Instances m_Instances;

  /** Instances with labels */
  protected Instances m_TotalTrainWithLabels;


  /** adjacency list for random */
  protected HashSet[] m_AdjacencyList;

  /** neighbor list for active learning: points in each cluster neighborhood */
  protected HashSet[] m_NeighborSets;

  /**
   * holds the global centroids
   */
  protected Instance m_GlobalCentroid;

  /**
   * holds the default perturbation value for randomPerturbInit
   */
  protected double m_DefaultPerturb = 0.7;

  protected boolean m_verbose = false;
  
  /** colors for DFS */
  final int WHITE = 300;
  final int GRAY = 301;
  final int BLACK = 302;

  /** number of neighborhood sets */
  protected int m_numNeighborhoods;


  /** Default constructors */
  public WeightedFFNeighborhoodInit() {
    super();
  } 

  /** Initialize with a clusterer */
  public WeightedFFNeighborhoodInit (MPCKMeans clusterer) {
    super(clusterer);
  }


  /** The main method for initializing cluster centroids
   */
  public Instances initialize() throws Exception {
    System.out.println("Num clusters = " + m_numClusters);
    m_Instances = m_clusterer.getInstances();
    m_TotalTrainWithLabels = m_clusterer.getTotalTrainWithLabels();
    m_ConstraintsHash = m_clusterer.getConstraintsHash();
    m_instanceConstraintHash = m_clusterer.getInstanceConstraintsHash();
    m_SeedHash = m_clusterer.getSeedHash();
    m_Seedable = m_clusterer.getSeedable();
    m_metric = m_clusterer.getMetric();
    m_RandomNumberGenerator = m_clusterer.getRandomNumberGenerator();
    m_objFunDecreasing = m_clusterer.getMetric().isDistanceBased();    

    m_NeighborSets = new HashSet[m_Instances.numInstances()];
    m_AdjacencyList = new HashSet[m_Instances.numInstances()];
    m_ClusterAssignments = new int [m_Instances.numInstances()];

    boolean m_isOfflineMetric = m_clusterer.getIsOfflineMetric();
    Instances m_ClusterCentroids = m_clusterer.getClusterCentroids();
    boolean m_useTransitiveConstraints = m_clusterer.getUseTransitiveConstraints();
    boolean m_isSparseInstance = (m_Instances.instance(0) instanceof SparseInstance) ? 
      true: false;
    if (m_isSparseInstance) {
      m_SumOfClusterInstances = new SparseInstance[m_Instances.numInstances()];
    } else {
      m_SumOfClusterInstances = new Instance[m_Instances.numInstances()];
    }
    
    for (int i=0; i<m_Instances.numInstances(); i++) {
      m_ClusterAssignments[i] = -1;
    }

    if (m_ConstraintsHash != null) {
      Set pointPairs = (Set) m_ConstraintsHash.keySet(); 
      Iterator pairItr = pointPairs.iterator();
      System.out.println("In non-active init");
      
      // iterate over the pairs in ConstraintHash
      while( pairItr.hasNext() ){
	InstancePair pair = (InstancePair) pairItr.next();
	int linkType = ((Integer) m_ConstraintsHash.get(pair)).intValue();
	if (m_verbose)
	  System.out.println(pair + ": type = " + linkType);
	if( linkType == InstancePair.MUST_LINK ){ // mainly concerned with MUST-LINK
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
      
      // DFS for finding connected components, updates requires stats
      DFS();
    }
    
    // print out cluster assignments right here!!
    if (m_ConstraintsHash.size() > 0) { 
      if (m_metric instanceof BarHillelMetric) {
	System.out.println("Starting building BarHillel metric ...\n\n");
	((BarHillelMetric) m_metric).buildAttributeMatrix(m_Instances, m_ClusterAssignments);
	System.out.println("Finished building BarHillel metric!!\n\n");
      } else if (m_metric instanceof XingMetric) {
	((XingMetric) m_metric).buildAttributeMatrix(m_Instances, m_ConstraintsHash);
      } else if (m_metric instanceof BarHillelMetricMatlab) {
	System.out.println("Starting building BarHillelMatlab metric ...\n\n");
	((BarHillelMetricMatlab) m_metric).buildAttributeMatrix(m_Instances, m_ClusterAssignments);
	System.out.println("Finished building BarHillelMatlab metric!!\n\n");
      }
    }
    
    if (!m_Seedable) { // don't perform any seeding, initialize from random
      m_NumCurrentClusters = 0;
      System.out.println("Not performing any seeding!");
      for (int i=0; i<m_Instances.numInstances(); i++) {
	m_ClusterAssignments[i] = -1;
      }
    }
    // if the required number of clusters has been obtained, wrap-up
    if( m_NumCurrentClusters >= m_numClusters ){
      {//if (m_verbose) {
	System.out.println("Got the required number of clusters ...");
	System.out.println("num clusters: " + m_numClusters + ", num current clusters: " + m_NumCurrentClusters);
      }

      int clusterSizes[] = new int[m_NumCurrentClusters];
      for (int i=0; i<m_NumCurrentClusters; i++) {
	if (m_verbose) {
	  System.out.println("Neighbor set: " + i + " has size: " + m_NeighborSets[i].size());
	}
	clusterSizes[i] = -m_NeighborSets[i].size(); // for reverse sort
      }	
      int[] indices = Utils.sort(clusterSizes);
      System.out.println("Total neighborhoods:  " + m_NumCurrentClusters + ";  Sorted neighborhood sizes:  ");

      // store number of neighborhoods after DFS
      m_numNeighborhoods = m_NumCurrentClusters;

      for (int i=0; i < m_NumCurrentClusters; i++) {
	System.out.print(m_NeighborSets[indices[i]].size());
	if (m_TotalTrainWithLabels.classIndex() >= 0) {
	  System.out.println("(" + m_TotalTrainWithLabels.instance(((Integer) (m_NeighborSets[indices[i]].iterator().next())).intValue()).classValue()+ ")\t");
	} else {
	  System.out.println();
	}
      }
      
      Instance[] clusterCentroids = new Instance[m_NumCurrentClusters];
      
      // Added: Code for better random selection of neighborhoods, using weighted farthest first
      for (int i=0; i<m_NumCurrentClusters; i++) { 
	if (m_isSparseInstance) {
	  clusterCentroids[i] = new SparseInstance(m_SumOfClusterInstances[i]);
	}
	else {
	  clusterCentroids[i] = new Instance(m_SumOfClusterInstances[i]);
	}
	clusterCentroids[i].setWeight(m_NeighborSets[i].size()); // setting weight = neighborhood size
	clusterCentroids[i].setDataset(m_Instances);
	if (!m_objFunDecreasing) {
	  ClusterUtils.normalize(clusterCentroids[i]);
	} else {
	  ClusterUtils.normalizeByWeight(clusterCentroids[i]);
	}
      }

      HashSet selectedNeighborhoods = new HashSet((int) (m_numClusters/0.75 + 10));
      System.out.println("Initializing " + m_numClusters + " clusters");
      if (m_isOfflineMetric) {
	System.out.println("Offline metric - using random neighborhoods");
	for (int i=0; i<m_numClusters; i++) {
	  int next = m_RandomNumberGenerator.nextInt(m_numNeighborhoods);
	  while (selectedNeighborhoods.contains(new Integer (next))) {
	    next = m_RandomNumberGenerator.nextInt(m_numNeighborhoods);
	  }
	  System.out.print("Neighborhood selected:  " + next);
	  if (m_TotalTrainWithLabels.classIndex() >= 0) {
	    System.out.println("(" + m_TotalTrainWithLabels.instance(((Integer)(m_NeighborSets[next].iterator().next())).intValue()).classValue()+ ")\t");
	  } else {
	    System.out.println();
	  }
	  selectedNeighborhoods.add(new Integer(next));	
	}
      } else {
	System.out.println("Learnable metric - using weighted FF to select neighborhoods");
	selectedNeighborhoods.add(new Integer(indices[0])); // initializing with largest neighborhood
	System.out.print("First neighborhood selected:  " + m_NeighborSets[indices[0]].size());
	if (m_TotalTrainWithLabels.classIndex() >= 0) {
	  System.out.println("(" + m_TotalTrainWithLabels.instance(((Integer)(m_NeighborSets[indices[0]].iterator().next())).intValue()).classValue()+ ")\t");
	} else {
	  System.out.println();
	}
	
	HashSet selectedNeighborhood = new HashSet();
	System.out.println("Initializing rest by weightedFarthestFromSetOfPoints");
	for (int i=1; i<m_numClusters; i++) {
	  int next = (int) weightedFarthestFromSetOfPoints(clusterCentroids, selectedNeighborhoods, null);
	  selectedNeighborhoods.add(new Integer(next));
	  System.out.print("Neighborhood selected:  " + m_NeighborSets[next].size());
	  if (m_TotalTrainWithLabels.classIndex() >= 0) {
	    System.out.println("(" + m_TotalTrainWithLabels.instance(((Integer)(m_NeighborSets[next].iterator().next())).intValue()).classValue()+ ")\t");
	  } else {
	    System.out.println();
	  }
	}
      }

      // compute centroids of m_numClusters clusters from selectedNeighborhoods
      m_ClusterCentroids = new Instances(m_Instances, m_numClusters);

      Iterator neighborhoodIter = selectedNeighborhoods.iterator(); 
      int num=0; // cluster number
      while (neighborhoodIter.hasNext()) {
	int i = ((Integer) neighborhoodIter.next()).intValue();
	if (m_SumOfClusterInstances[i] != null) {
	  if (m_verbose) {
	    System.out.println("Normalizing instance " + i);
	  }
	  if (!m_objFunDecreasing) {
	    ClusterUtils.normalize(m_SumOfClusterInstances[i]);
	  }
	  else {
	    ClusterUtils.normalizeByWeight(m_SumOfClusterInstances[i]);
	  }
	}
	Iterator iter = m_NeighborSets[i].iterator();
	while (iter.hasNext()) { // assign points of new cluster
	  int instNumber = ((Integer) iter.next()).intValue();
	  if (m_verbose) {
	    System.out.println("Assigning " + instNumber + " to cluster: " + num);
	  }
	  m_ClusterAssignments[instNumber] = num;
	}

	m_SumOfClusterInstances[num].setDataset(m_Instances);
	m_ClusterCentroids.add(m_SumOfClusterInstances[i]);
	num++;
      }
      for (int j=0; j < m_NumCurrentClusters; j++) {
	int i = indices[j];
	if (!selectedNeighborhoods.contains(new Integer(i))) { // not assigned as centroid
	  Iterator iter = m_NeighborSets[i].iterator();
	  while (iter.hasNext()) {
	    int instNumber = ((Integer) iter.next()).intValue();
	    if (m_verbose) {
	      System.out.println("Assigning " + instNumber + " to cluster -1");
	    }
	    m_ClusterAssignments[instNumber] = -1;
	  }
	}
      }

      m_NumCurrentClusters = m_numClusters;

      // adding other inferred ML and CL links
      if (m_useTransitiveConstraints) { 
	addMLAndCLTransitiveClosure(indices);
	System.out.println("Adding constraints by transitive closure");
      } else {
	  System.out.println("Not adding constraints by transitive closure");
      }
    } else if( m_NumCurrentClusters < m_numClusters ){
      // make random for rest

      // adding other inferred ML and CL links
      if (m_useTransitiveConstraints) { 
	addMLAndCLTransitiveClosure(null);
      }
      
      System.out.println("Found " + m_NumCurrentClusters + " neighborhoods ...");
      System.out.println("Will have to start " + (m_numClusters - m_NumCurrentClusters) + " clusters at random");
	
      // compute centroids of m_NumCurrentClusters clusters
      m_ClusterCentroids = new Instances(m_Instances, m_numClusters);
      for (int i=0; i<m_NumCurrentClusters; i++) {
	if (m_SumOfClusterInstances[i] != null) {
	  if (m_verbose) {
	    System.out.println("Normalizing cluster center " + i);
	  }
	  if (!m_objFunDecreasing) {
	    ClusterUtils.normalize(m_SumOfClusterInstances[i]);
	  } else {
	    ClusterUtils.normalizeByWeight(m_SumOfClusterInstances[i]);
	  }
	}
	m_SumOfClusterInstances[i].setDataset(m_Instances);
	m_ClusterCentroids.add(m_SumOfClusterInstances[i]);
      }

      // find global centroid
      double [] globalValues = new double[m_Instances.numAttributes()];
      if (m_isSparseInstance) {
	globalValues = ClusterUtils.meanOrMode(m_Instances); // uses fast meanOrMode
      } else {
	for (int j = 0; j < m_Instances.numAttributes(); j++) {
	  globalValues[j] = m_Instances.meanOrMode(j); // uses usual meanOrMode
	}
      }
      
      System.out.println("Done calculating global centroid");
      // global centroid is dense in SPKMeans
      m_GlobalCentroid = new Instance(1.0, globalValues);
      m_GlobalCentroid.setDataset(m_Instances);
      if (!m_objFunDecreasing) {
	ClusterUtils.normalizeInstance(m_GlobalCentroid);
      }
      //      System.out.println("Global centroid is: " + m_GlobalCentroid);

      for (int i=m_NumCurrentClusters; i<m_numClusters; i++) {
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
	
	if (m_isSparseInstance) {
	  m_ClusterCentroids.add(new SparseInstance(1.0, values)); // sparse for consistency with other cluster centroids
	} else {
	  m_ClusterCentroids.add(new Instance(1.0, values));
	}
      }
      System.out.println("Done calculating random centroids by perturbation");
      m_NumCurrentClusters = m_numClusters;
    }

    m_clusterer.setClusterAssignments(m_ClusterAssignments);
    return m_ClusterCentroids;
  }



  /** Main Depth First Search routine */
  protected void DFS() throws Exception {
    int [] vertexColor = new int[m_Instances.numInstances()];
    m_NumCurrentClusters = 0;

    for(int u=0; u<m_Instances.numInstances(); u++){
      vertexColor[u] = WHITE;
    }
    
    for(int u=0; u<m_Instances.numInstances(); u++){
      // NOTE: Have to uncomment check for m_AdjacencyList != null to enable farthestFirst
      // as default initialization, instead of randomPerturbInit
      if (m_AdjacencyList[u] != null && vertexColor[u] == WHITE) {
	m_NeighborSets[m_NumCurrentClusters] = new HashSet();
	//  	m_NeighborSets[m_NumCurrentClusters].add(new Integer(u));
	//  	m_SumOfClusterInstances[m_NumCurrentClusters] = sumWithInstance(m_SumOfClusterInstances[m_NumCurrentClusters], m_Instances.instance(u));
	//  	m_ClusterAssignments[u] = m_NumCurrentClusters;
	DFS_VISIT(u, vertexColor); 	// found whole neighbourhood of u
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
    m_SumOfClusterInstances[m_NumCurrentClusters] = ClusterUtils.sumWithInstance(m_SumOfClusterInstances[m_NumCurrentClusters], m_Instances.instance(u),m_Instances);
    vertexColor[u] = BLACK;
  }


  /** Finds point in setOfPoints which has weighted max min-distance from set visitedPoints
   */
  int weightedFarthestFromSetOfPoints(Instance[] setOfPoints, HashSet visitedPoints, HashSet eliminationSet) 
    throws Exception {
    
    // implements weighted farthest-first search algorithm in the given setOfPoints:
    /*
      for (each datapoint x from setOfPoints not in visitedPoints) {
      distance of x to visitedPoints = min{weighted d(x,f):f \in visitedPoints}
      }
      select the point x with maximum distance as new center;
    */

    if (visitedPoints.size() == 0) {
      int point;
      point = m_RandomNumberGenerator.nextInt(setOfPoints.length);
      // Note - no need to check for labeled data now, since we have
      // no visitedPoints => no labeled data
      if (m_verbose)
	System.out.println("First point selected: " + point);
      return point;
    }
    
    double minSimilaritySoFar = Double.POSITIVE_INFINITY;
    double maxDistanceSoFar = Double.NEGATIVE_INFINITY;
    ArrayList bestPoints = new ArrayList();

    for (int i=0; i<setOfPoints.length; i++) {
      if (!visitedPoints.contains(new Integer(i))) {
	if (eliminationSet == null || !eliminationSet.contains(new Integer(i))) {
	  // point should not belong to visitedPoints or eliminationSet
	  Instance inst = setOfPoints[i];
	  Iterator iter = visitedPoints.iterator();
	  double minDistanceFromSet = Double.POSITIVE_INFINITY;
	  double maxSimilarityFromSet = Double.NEGATIVE_INFINITY;
	  while (iter.hasNext()) {
	    Instance pointInSet = setOfPoints[((Integer) iter.next()).intValue()];
	    if (!m_objFunDecreasing) {
	      double sim = m_metric.similarity(inst, pointInSet) / Math.sqrt(pointInSet.weight() * inst.weight());
	      if (sim > maxSimilarityFromSet) {
		maxSimilarityFromSet = sim;
	      }
	    } else {
	      double dist = 0; 
	      if (m_metric instanceof KL) {
		dist = ((KL)m_metric).distanceJS(inst, pointInSet) * Math.sqrt(pointInSet.weight() * inst.weight());
	      } else {
		dist = m_metric.distance(inst, pointInSet) * Math.sqrt(pointInSet.weight() * inst.weight());
	      } 
	      if (dist < minDistanceFromSet) {
		minDistanceFromSet = dist;
	      }
	    }
	  }
	  if (!m_objFunDecreasing) {
	    if (maxSimilarityFromSet == minSimilaritySoFar) {
	      minSimilaritySoFar = maxSimilarityFromSet;
	      bestPoints.add(new Integer(i));
	    } else if (maxSimilarityFromSet < minSimilaritySoFar) {
	      minSimilaritySoFar = maxSimilarityFromSet;
	      bestPoints.clear();
	      bestPoints.add(new Integer(i));
	    }
	  } else {
	    if (minDistanceFromSet == maxDistanceSoFar) {
	      minDistanceFromSet = maxDistanceSoFar;
	      bestPoints.add(new Integer(i));
	      if (m_verbose) {
		System.out.println("Additional point added: " + i + " with distance: " + maxDistanceSoFar);
	      }
	    } else if (minDistanceFromSet > maxDistanceSoFar) {
	      maxDistanceSoFar = minDistanceFromSet;
	      bestPoints.clear();
	      bestPoints.add(new Integer(i));
	      if (m_verbose) {
		System.out.println("Farthest point from set is: " + i + " with distance: " + maxDistanceSoFar);
	      }
	    }
	  }
	}
      }
    }

    int bestPoint = -1;
    if (bestPoints.size() > 1) { // multiple points, get random from whole set
      bestPoint = m_RandomNumberGenerator.nextInt(setOfPoints.length);

      while ((visitedPoints != null && visitedPoints.contains(new Integer(bestPoint)))
	     || (eliminationSet != null && eliminationSet.contains(new Integer(bestPoint)))) {
	bestPoint = m_RandomNumberGenerator.nextInt(setOfPoints.length);
      }
    }
    else { // only 1 point, fine
      bestPoint = ((Integer)bestPoints.get(0)).intValue();
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
	      InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
	      if (!m_ConstraintsHash.containsKey(pair)) {
		m_ConstraintsHash.put(pair, new Integer(InstancePair.MUST_LINK));
		if (m_verbose) {
		  System.out.println("Adding inferred ML (" + pair.first +","+pair.second+")");
		}
		
		// hash the constraints for the instances involved
		Integer firstInt = new Integer(first);
		Integer secondInt = new Integer(second);
		InstancePair pairML = new InstancePair(first, second, InstancePair.MUST_LINK);

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
		  InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
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
		  else if (first < second) {  // add new constraint
		    InstancePair pair = new InstancePair(first, second, InstancePair.DONT_CARE_LINK);
		    if (!m_ConstraintsHash.containsKey(pair)) {
		      m_ConstraintsHash.put(pair, new Integer(InstancePair.CANNOT_LINK));
		      if (m_verbose) {
			System.out.println("Adding inferred CL (" + pair.first +","+pair.second+")");
		      }

		      // hash the constraints for the instances involved
		      Integer firstInt = new Integer(first);
		      Integer secondInt = new Integer(second);
		      InstancePair pairCL = new InstancePair(first, second, InstancePair.CANNOT_LINK);
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
    m_clusterer.setInstanceConstraintsHash(m_instanceConstraintHash);
  }


  public void setOptions (String[] options)
    throws Exception {
    // TODO
  }

  public Enumeration listOptions () {
    // TODO
    return null;
  }
  
  public String [] getOptions ()  {
    String[] options = new String[10];
    int current = 0;

    options[current++] = "-N";
    options[current++] = "" + m_numClusters;

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }
}
