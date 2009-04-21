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
 *    MPCKMeansInitializer.java
 *    An abstract class for algorithms that do initialization
 *    Copyright (C) 2004 Sugato Basu
 *
 */

package weka.clusterers.initializers; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.clusterers.*;

public abstract class MPCKMeansInitializer implements Cloneable, Serializable, OptionHandler {
  /** Clusterer that the initializer operates on */
  protected MPCKMeans m_clusterer = null;

  /** Number of Clusters */
  protected int m_numClusters;

  /** Default constructors */
  public MPCKMeansInitializer() {
  } 

  /** Initialize with a clusterer */
  public MPCKMeansInitializer (MPCKMeans clusterer) {
    setClusterer(clusterer);
  }

  /** Set the clusterer */
  public void setClusterer(MPCKMeans clusterer) {
    this.m_clusterer = clusterer;
  }

  public void setNumClusters(int numClusters) {
    m_numClusters = m_clusterer.getNumClusters();
    System.out.println("Initializing " + m_numClusters + " clusters");
  }

  /** The main method
   *
   * @return the cluster centroids after initialization
   */
  public abstract Instances initialize() throws Exception;
} 

