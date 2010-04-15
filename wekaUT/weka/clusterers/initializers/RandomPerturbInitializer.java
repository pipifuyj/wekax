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
 *    RandomPerturbInitializer.java
 *
 *    Initializer that perturbs the global centroid randomly to get
 *    initial clusters for K-Means
 *
 *    Copyright (C) 2004 Sugato Basu, Misha Bilenko
 * 
 */

package weka.clusterers.initializers; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.clusterers.*;

public class RandomPerturbInitializer extends MPCKMeansInitializer {
  /** Default perturbation */
  protected double m_DefaultPerturb = 0.7;

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


  /** Default constructors */
  public RandomPerturbInitializer() {
    super();
  } 

  /** Initialize with a clusterer */
  public RandomPerturbInitializer (MPCKMeans clusterer) {
    super(clusterer);
  }

  /** The main method for initializing cluster centroids
   *
   * @return the cluster centroids after initialization
   */
  public Instances initialize() throws Exception {
    System.out.println("Num clusters = " + m_numClusters);
    Instances m_Instances = m_clusterer.getInstances();
    boolean m_useTransitiveConstraints = m_clusterer.getUseTransitiveConstraints();
    Instances m_ClusterCentroids = m_clusterer.getClusterCentroids();
    boolean m_objFunDecreasing = m_clusterer.getMetric().isDistanceBased();
    Random m_RandomNumberGenerator = m_clusterer.getRandomNumberGenerator();
    boolean m_isSparseInstance = (m_Instances.instance(0) instanceof SparseInstance) ? 
      true: false;
    

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
    Instance m_GlobalCentroid = new Instance(1.0, globalValues);
    m_GlobalCentroid.setDataset(m_Instances);
    if (!m_objFunDecreasing) {
      ClusterUtils.normalizeInstance(m_GlobalCentroid);
    }
    
    for (int i=0; i<m_numClusters; i++) {
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
      }
      else {
	m_ClusterCentroids.add(new Instance(1.0, values));
      }
    }
    System.out.println("Initialized centroids by RandomPerturbGlobal");
    //    System.out.println("Centroids are: " + m_ClusterCentroids);

    return m_ClusterCentroids;
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
