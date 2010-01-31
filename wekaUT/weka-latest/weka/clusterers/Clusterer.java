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
 *    Clusterer.java
 *    Copyright (C) 1999 Mark Hall
 *
 */

package weka.clusterers;

import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.SerializedObject;
import weka.core.Utils;
import weka.core.metrics.*;


/** 
 * Abstract clusterer.
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision: 1.1.1.1 $
 */
public abstract class Clusterer implements Cloneable, Serializable {

  // ===============
  // Public methods.
  // ===============
 
  /**
   * Generates a clusterer. Has to initialize all fields of the clusterer
   * that are not being set via options.
   *
   * @param data set of instances serving as training data 
   * @exception Exception if the clusterer has not been 
   * generated successfully
   */
  public abstract void buildClusterer(Instances data) throws Exception;

  /**
   * Classifies a given instance.
   *
   * @param instance the instance to be assigned to a cluster
   * @return the number of the assigned cluster as an interger
   * if the class is enumerated, otherwise the predicted value
   * @exception Exception if instance could not be classified
   * successfully
   */
  public abstract int clusterInstance(Instance instance) throws Exception; 

  /**
   * Returns the number of clusters.
   *
   * @return the number of clusters generated for a training dataset.
   * @exception Exception if number of clusters could not be returned
   * successfully
   */
  public abstract int numberOfClusters() throws Exception;

  /**
   * Creates a new instance of a clusterer given it's class name and
   * (optional) arguments to pass to it's setOptions method. If the
   * clusterer implements OptionHandler and the options parameter is
   * non-null, the clusterer will have it's options set.
   *
   * @param searchName the fully qualified class name of the clusterer
   * @param options an array of options suitable for passing to setOptions. May
   * be null.
   * @return the newly created search object, ready for use.
   * @exception Exception if the clusterer class name is invalid, or the 
   * options supplied are not acceptable to the clusterer.
   */
  public static Clusterer forName(String clustererName,
				  String [] options) throws Exception {
    return (Clusterer)Utils.forName(Clusterer.class,
				    clustererName,
				    options);
  }

  /**
   * Creates copies of the current clusterer. Note that this method
   * now uses Serialization to perform a deep copy, so the Clusterer
   * object must be fully Serializable. Any currently built model will
   * now be copied as well.
   *
   * @param model an example clusterer to copy
   * @param num the number of clusterer copies to create.
   * @return an array of clusterers.
   * @exception Exception if an error occurs 
   */
  public static Clusterer [] makeCopies(Clusterer model,
					int num) throws Exception {
     if (model == null) {
      throw new Exception("No model clusterer set");
    }
    Clusterer [] clusterers = new Clusterer [num];
    SerializedObject so = new SerializedObject(model);
    for(int i = 0; i < clusterers.length; i++) {
      clusterers[i] = (Clusterer) so.getObject();
    }
    return clusterers;
  }
  
  /**
  Initializer need these methods, and these should be overwritten.
  */
  public Instances getInstances(){
      return new Instances("Instances",new FastVector(),0);
  }
  public Metric fetchMetric(){
      return new Euclidean();
  }
  public Instances getCluster(int clusterId) throws Exception{
      Instances instances=getInstances();
      Instances cluster=new Instances(instances,0);
      Instance instance;
      for(int i=0,ii=instances.numInstances();i<ii;i++){
          instance=instances.instance(i);
          if(clusterId==clusterInstance(instances.instance(i))){
              cluster.add(instance);
          }
      }
      return cluster;
  }
}


