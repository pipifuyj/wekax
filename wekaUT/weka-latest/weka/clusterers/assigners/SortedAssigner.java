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
 *    SortedAssigner.java
 *    Random order assignment for K-Means
 *    Copyright (C) 2004 Misha Bilenko, Sugato Basu
 *
 */

package weka.clusterers.assigners; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.clusterers.*;


public class SortedAssigner extends MPCKMeansAssigner {
  /** Move points in assignment step till stabilization? */
  protected boolean m_MovePointsTillAssignmentStabilizes = false;
  
  /** Number of times points are moved in assignment step till stabilization */
  protected int m_MaxTimesPointsMoved = 100;

  /** This is a sequential assignment method */
  public boolean isSequential() {
    return true;
  }

  /** The main method
   *  @return the number of points that changed assignment
   */
  public int assign() throws Exception {    
    int moved = 0;
    Instances instances = m_clusterer.getInstances();
    int numInstances = instances.numInstances();
    int numClusters = m_clusterer.getNumClusters();

    double bestDistance = Double.MAX_VALUE;
    double bestSimilarity = Double.MIN_VALUE;
    int [] sortedIndices = null;
    double [] distances = new double[numInstances];

    // find closest cluster centroid for each instance
    for (int i = 0; i < numInstances; i++) {
      for (int j = 0; j < numClusters; j++) {
	double distance = 0; 

	  // get the current distance
	distance = m_clusterer.penaltyForInstance(i,j);
	
	// update if best
	if (distance < bestDistance) {
	  bestDistance = distance; 
	  distances[i] = distance;
	}
      }
    }

    sortedIndices = Utils.sort(distances); // sort in ascending order
      
    for (int i=0; i < numInstances; i++) {
      // Update number of points moved
      moved += m_clusterer.assignInstanceToClusterWithConstraints(sortedIndices[i]);
    }

    if (m_MovePointsTillAssignmentStabilizes) {
      int newMoved = -1;
      for (int t=0; t<m_MaxTimesPointsMoved && newMoved != 0; t++) { // move points till assignment stabilizes
	newMoved = 0;
	m_clusterer.resetObjective();
	for (int i=0; i<numInstances; i++) {
	  newMoved += m_clusterer.assignInstanceToClusterWithConstraints(sortedIndices[i]);
	}
	if (newMoved > 0) {
	  System.out.println(newMoved + " points moved on changing order in t=" + t);
	} else {
	  break; // go out of for loop
	} 
      }
    }
    return moved;
  }

  /**
   * Get/Set m_MovePointsTillAssignmentStabilizes
   * @param b truth value
   */
  public void setMovePointsTillAssignmentStabilizes (boolean b) {
    this.m_MovePointsTillAssignmentStabilizes = b;
  }
  public boolean getMovePointsTillAssignmentStabilizes () {
    return  m_MovePointsTillAssignmentStabilizes;
  }

  /** Get/set the number of times points can be moved */
  public int getMaxTimesPointsMoved() {
    return m_MaxTimesPointsMoved;
  }
  public void setMaxTimesPointsMoved(int  v) {
    this.m_MaxTimesPointsMoved = v;
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
    String[] options = new String[20];
    int current = 0;

    if (m_MovePointsTillAssignmentStabilizes) {
      options[current++] = "-move";
      options[current++] = "" + getMaxTimesPointsMoved();
    }

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }
} 
