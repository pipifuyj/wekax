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
 *    MPCKMeansAssigner.java
 *    An abstract class for algorithms that do E-step assignment
 *    Copyright (C) 2004 Misha Bilenko
 *
 */

package weka.clusterers.assigners; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.clusterers.*;

public abstract class MPCKMeansAssigner implements Cloneable, Serializable, OptionHandler {
  /** Clusterer that the assigner operates on */
  protected MPCKMeans m_clusterer = null;

  /** Default constructors */
  public MPCKMeansAssigner() {
  } 

  /** Initialize with a clusterer */
  public MPCKMeansAssigner (MPCKMeans clusterer) {
    setClusterer(clusterer);
  }

  /** Set the clusterer */
  public void setClusterer(MPCKMeans clusterer) {
    this.m_clusterer = clusterer;
  }

  /** Assigners can be sequential or collective */
  public abstract boolean isSequential(); 

  /** The main method
   *  @return the number of points that changed assignment
   */
  public abstract int assign() throws Exception;

} 

