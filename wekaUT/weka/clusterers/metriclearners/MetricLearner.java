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
 *    MetricLearner.java
 *    Copyright (C) 2004 Mikhail Bilenko and Sugato Basu
 *
 */

package weka.clusterers.metriclearners; 


import weka.core.metrics.LearnableMetric;
import weka.clusterers.SemiSupClusterer;

import java.io.Serializable;

/** 
 * A parent class for MPCKMeans metric learners
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu
 * @version $Revision: 1.4 $ */

public abstract class MetricLearner implements Cloneable, Serializable {

  /** The clusterer that the learner is attached to */
  protected SemiSupClusterer m_clusterer = null;
  public void setClusterer(SemiSupClusterer clusterer) {
    m_clusterer = clusterer;
  }
  public SemiSupClusterer getClusterer() {
    return m_clusterer;
  }


  /** The metric being learned */
  protected LearnableMetric m_metric = null;
  public void setMetric(LearnableMetric metric) {
    m_metric = metric;
  }
  public LearnableMetric getMetric() {
    return m_metric;
  }

  /** The most important method */
  public abstract boolean trainMetric(int clusterIdx) throws Exception;

  /** Create a copy of this metric  learner*/
  public Object clone() {
    MetricLearner ml = null; 
    try {
      ml = (MetricLearner) super.clone();
    } catch (CloneNotSupportedException e) {
      System.err.println("Metric learner can't clone");
    }
    // copy the fields
    ml.setClusterer(this.m_clusterer);
    ml.setMetric(this.m_metric); 
    return ml;
  } 
}






