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
 *    SemiSupClusterer.java
 *    Copyright (C) 2001 Mikhail Bilenko, Sugato Basu
 *
 */


/**
 * Semi-Supervised Clusterer interface.
 *
 * @version $Revision: 1.5 $
 */
package weka.clusterers;


import java.util.HashMap;
import java.util.ArrayList;

import weka.core.Instances;
import weka.core.metrics.LearnableMetric;

public interface SemiSupClusterer {
  /**
   * We always want to implement SemiSupClusterer from a class extending Clusterer.  
   * We want to be able to return the underlying parent class.
   * @return parent Clusterer class
   */
  abstract Clusterer getThisClusterer();
  
  /**
   * Set the number of clusters.
   */
  abstract void setNumClusters (int n);

  /**
   * Get the number of clusters.
   */
  abstract int getNumClusters ();
  
  /**
   * Sets verbose level
   */
  abstract void setVerbose (boolean v);
 
  /**
   * Returns an ArrayList of clusters
   */
  abstract ArrayList getClusters() throws Exception;
  
 /**
   * Return the instances used for clustering
   *
   * @return Instances used for clustering, or null 
   */
  abstract Instances getInstances() throws Exception;
  
  /**
   * Generates the clustering. 
   *
   * @param data set of instances to cluster
   * @exception Exception if something is wrong
   */
  abstract void buildClusterer (Instances data) throws Exception;
  
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
  abstract void buildClusterer (Instances labeledData, Instances unlabeledData, int classIndex, int numClusters, int startingIndexOfTest) throws Exception;

  /**
   * Train the clusterer using provided training data
   *
   * @param instnaces instances to be used for training
   */
  abstract void trainClusterer (Instances instances) throws Exception;
  
  /**
   * Seed the clusterer using specified seeding
   *
   * @param seed_params HashMap of seeding parameters
   */
  abstract void seedClusterer (HashMap seed_params) throws Exception;
        
  /**
   * Reset all values that have been learned
   */
  abstract void resetClusterer() throws Exception;
  
  /**
   * Set the clusterer metric 
   */
  abstract void setMetric (LearnableMetric m) throws Exception;

  /** 
   *  Returns objective function if it has one, else -1.
   *  Needed for SemiSupClustererEvaluation.
   */
  abstract double objectiveFunction();
}
