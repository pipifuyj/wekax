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
 *    Cluster.java
 *    Copyright (C) 2001 Mikhail Bilenko, Sugato Basu
 *
 */

package weka.clusterers;

import weka.core.Instance;
import java.util.Enumeration;
import java.util.ArrayList;

public class Cluster extends ArrayList {
  /** contains weights corresponding to each cluster member, 
      corresponds to probabilities for belonging to clusters */
  ArrayList weights;

  /** corresponds to a cluster ID number, useful for identifying the cluster */
  public int clusterID;

  /* Unused now, could be used later 
     protected int type; 
  */
  
  /**
   * Creates an empty cluster
   */
  public Cluster() {
    super();
    weights = new ArrayList();
  }

  /**
   * Creates an empty cluster with an id
   */
  public Cluster(int id) {
    super();
    weights = new ArrayList();
    clusterID = id;
  }

  /**
   * Creates a singleton cluster, assignes weight 1 to the instance
   *
   * @param o object to be added to the cluster
   */
  public Cluster (Object o) {
    this(o, 1.0);
  }
    
  /**
   * Creates a singleton cluster, assignes specified weight to the instance
   *
   * @param o object to be added to the cluster
   * @param wt weight of the object
   */
  public Cluster (Object o, double wt) {
    this();
    add (o, wt);
  }
  
  /**
   * Adds an object to the cluster with default weight 1
   *
   * @param o object to be added to the cluster
   */
  public boolean add (Object o) {
    add (o, 1.0);
    return true;
  }
  
  /**
   * Adds an object to the cluster with specified weight
   *
   * @param o object to be added to the cluster
   * @param wt weight of the object
   */
  public boolean add (Object instance, double wt) {
    super.add (instance);
    weights.add (new Double(wt));
    return true;
  }
  
  /**
   * Returns the weight of the element at the given position.
   *
   * @param index the element's index
   * @return the weight of the element with the given index
   */
  public double weightAt (int index) throws Exception {
    if (index < size()) 
      return ((Double)weights.get(index)).doubleValue();
    else
      throw new Exception ("Element index too large");
  }
  
  
  /**
   * Adds elements from another cluster to this cluster.  Weights are
   * kept the same
   *
   * @param old_cluster a Cluster containing elements to be added
   */ 
  public void copyElements(Cluster old_cluster) throws Exception {
    for (int i = 0; i < old_cluster.size(); i++) {
      add(old_cluster.get(i), old_cluster.weightAt(i));
    }
  }
		
  public String toString() {
    String s = "[";
    String c[];
    String[][] array = (String[][]) toArray();
    for (int i=0; i<array.length; i++){
      c = array[i];
      s = s + " " + c[0];
    }
    s = s + "]";
    return s;
  }
}
