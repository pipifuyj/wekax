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

import java.util.*;

import weka.core.*;
import weka.core.metrics.LearnableMetric;
import weka.clusterers.MPCKMeans;


/** 
 * A parent class for MPCKMeans metric learners
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu
 * @version $Revision: 1.12 $ */

public abstract class MPCKMeansMetricLearner extends  MetricLearner
  implements OptionHandler { 
  /** The number of attributes */
  protected int m_numAttributes = 0; 

  /** The clusterer */
  protected MPCKMeans m_kmeans = null;

  /** Current instances being clustered */
  protected Instances m_instances = null;

  /** Cluster assignments */
  protected int[] m_clusterAssignments = null;

  /** Map from instanceIdx to a list of constraints */
  protected HashMap m_instanceConstraintMap = null;

  /** Term weights */
  protected double m_MLweight = 0;
  protected double m_CLweight = 0;
  protected double m_logTermWeight = 0;
  protected double m_regularizerTermWeight = 0; 

  /** Centroid can be memorized for speedup if a metric for a single
      cluster is being trained */
  protected Instance m_centroid = null;

  /** Same for maxCLDiffInstance */
  protected Instance m_maxCLDiffInstance = null;

  /** Convergence difference for Newton-Raphson */
  protected double m_convergenceDifference = 1e-5;

  /** Minimum weight below which no weight can go */
  protected double m_minWeightValue = 1e-5; 

  /** Regularization */
  protected boolean m_regularize = false; 

  /** Initialize all the fields that will be used by individual learners */
  protected void Init(int clusterIdx) {
    m_kmeans = (MPCKMeans) m_clusterer;
    m_instances = m_kmeans.getInstances();
    m_numAttributes = m_instances.numAttributes();
    m_clusterAssignments = m_kmeans.getClusterAssignments();
    m_instanceConstraintMap = m_kmeans.getInstanceConstraintsHash();
    m_MLweight = m_kmeans.getMustLinkWeight();
    m_CLweight = m_kmeans.getCannotLinkWeight();
    m_logTermWeight = m_kmeans.getLogTermWeight();
    m_regularizerTermWeight = m_kmeans.getRegularizerTermWeight();

    m_regularize = m_kmeans.getRegularize();

    if (clusterIdx >= 0) {
      m_centroid = m_kmeans.getClusterCentroids().instance(clusterIdx);
    }

    if (m_instanceConstraintMap != null && m_instanceConstraintMap.size() > 0) {
      if (clusterIdx < 0) {
	m_maxCLDiffInstance = m_kmeans.m_maxCLDiffInstances[0];
      } else { 
	m_maxCLDiffInstance = m_kmeans.m_maxCLDiffInstances[clusterIdx];
      }
    }
  }

  public void resetLearner() {
  }

    /** calculates weights using Newton Raphson, to satisfy the
      positivity constraint of each attribute weight, returns learned
      attribute weights. Note: currentAttrWeights is the inverted version
      of the current metric weights.
  */
  protected double [] updateWeightsUsingNewtonRaphson
    (double [] currentAttrWeights, double [] invUnconstrainedAttrWeights)
    throws Exception {
    double [] iterAttrWeights = currentAttrWeights;
    double oldObjective, newObjective; 
    
    System.out.println("Updating Weights Using NewtonRaphson");
    do {
      // sets new attribute weights using NR with line search for alpha
      iterAttrWeights = nrWithLineSearchForAlpha(iterAttrWeights,
						 invUnconstrainedAttrWeights); 
      // set current attribute weight to metric, recalculate obj. fn.
      oldObjective = m_kmeans.objectiveFunction();
      m_metric.setWeights(iterAttrWeights);
      newObjective = m_kmeans.calculateObjectiveFunction(false);
    } while (!m_kmeans.convergenceCheck(oldObjective,
			       newObjective));
    // objective function not guaranteed to monotonically decrease
    // across NR iterations, so don't do convergence check
    return iterAttrWeights;
  }

  /** Does one NR step, calculates the alpha (using line search) that
      does not violate positivity constraint of each attribute weight,
      returns new values of attribute weights */
  protected double [] nrWithLineSearchForAlpha
    (double [] currAttrWeights, double [] invUnconstrainedAttrWeights)
    throws Exception {
    double [] raphsonWeights = new double[m_numAttributes];
    double top = 1, bottom = 0, alpha = 1;
    boolean satisfiesConstraints = true;
        
    // initial check for alpha = top
    System.out.println("Evaluating at alpha=1");
    for (int attr = 0; attr < m_numAttributes; attr++) {
      raphsonWeights[attr] = currAttrWeights[attr] * (1 - alpha * (currAttrWeights[attr] * invUnconstrainedAttrWeights[attr] - 1));
      if (raphsonWeights[attr] < 0) {
	satisfiesConstraints = false;
	System.out.println("Negative raphsonWeight for attr: " + attr + ", exiting loop");
	break;
      }
      //        System.out.println("Curr weights: " + currAttrWeights[attr] + ", alpha: " + alpha + ", m_Objective: " + m_Objective);
      //        System.out.println("Raphson weights[" + attr +"] = " + raphsonWeights[attr]);
    }

    if (!satisfiesConstraints) {
      // line search for alpha between bottom and top
      // satisfiesConstraints is false at top, true at bottom
      // we want max. alpha in [0,1] for which satisfiesConstraints is true
      System.out.println("Starting line search for alpha");
      while ((top-bottom) > m_convergenceDifference && bottom <= top) {
	alpha = (bottom + top)/2;
	satisfiesConstraints = true;
	for (int attr = 0; attr < m_numAttributes; attr++) {
	  raphsonWeights[attr] = currAttrWeights[attr] * (1 - alpha * (currAttrWeights[attr] * invUnconstrainedAttrWeights[attr] - 1));
	  if (raphsonWeights[attr] < 0) {
	    satisfiesConstraints = false;
	    System.out.println("Negative raphsonWeight for attr: " + attr + ", exiting loop");
	    break;
	  }
	  //  	  System.out.println("In line search ... curr weights: " + currAttrWeights[attr] + ", alpha: " + alpha + ", m_Objective: " + m_Objective);
	  //  	  System.out.println("In line search ... raphson weights[" + attr +"] = " + raphsonWeights[attr]);
	}
	if (!satisfiesConstraints) {
	  top = alpha;
	} else {
	  bottom = alpha;
	}
	System.out.println("Top: " + top + ", Bottom: " + bottom);
      }
      alpha = bottom;
    
      System.out.println("Final alpha: " + alpha);
      System.out.print("Final weights: ");
      for (int attr = 0; attr < m_numAttributes; attr++) {
	raphsonWeights[attr] = currAttrWeights[attr] * (1 - alpha * (currAttrWeights[attr] * invUnconstrainedAttrWeights[attr] - 1));
	System.out.print(raphsonWeights[attr] + "\t");
      }
      System.out.println();
    } else {
      System.out.println("Constraints satisfied");
    }

    return raphsonWeights;
  }

  abstract public String [] getOptions();
  abstract public void setOptions(String[] options) throws Exception;
  abstract public Enumeration listOptions();


  public void setMinWeightValue(double min) {
    m_minWeightValue = min;
  }
  public double getMinWeightValue() {
    return m_minWeightValue;
  } 
}
