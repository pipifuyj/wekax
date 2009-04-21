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
 * @version $Revision: 1.2 $ */

public abstract class GDMetricLearner extends  MPCKMeansMetricLearner {
  /** Initial value of gradient descent step parameter */
  protected double m_eta = 0.001;
  public void setEta(double eta) {
    m_eta = eta;
  }

  public void resetLearner() {
    m_currEta = m_eta;
  } 

  public double getEta() {
    return m_eta;
  }

  /** Current value of the step parameter */
  protected double m_currEta = 0; 

  /** Decay rate of gradient descent eta */
  protected double m_etaDecayRate = 0.9;  
  public void setEtaDecayRate(double etaDecayRate) {
    m_etaDecayRate = etaDecayRate;
  }
  public double getEtaDecayRate() {
    return m_etaDecayRate;
  }

  /** The maximum number of GD iterations */
  protected int m_maxGDIterations = 20;
  public void setMaxGDIterations(int maxGDIterations) {
    m_maxGDIterations = maxGDIterations;
  }
  public int getMaxGDIterations() {
    return m_maxGDIterations;
  }


  protected double[] InitRegularizerComponents(double []currentWeights) {
    double [] regularizerComponents = new double[m_numAttributes];
    for (int attr = 0; attr < m_numAttributes; attr++) {
      if (currentWeights[attr] > 0) {
	regularizerComponents[attr] = m_regularizerTermWeight
	  * m_metric.getRegularizer().gradient(currentWeights[attr]);
      } else {
	regularizerComponents[attr] = 0;
      }
    }

    return regularizerComponents;
  }



  /**
   * Perform gradient step update using the current weights,
   * the gradients, the regularizers and the current learning rate
   * Returns the updated weights.
   **/ 
  protected double[] GDUpdate(double [] currentWeights,
			      double [] gradients,
			      double [] regularizerComponents) {
    double [] newWeights = new double[m_numAttributes]; 

    for (int attr = 0; attr < m_numAttributes; attr++) {
      newWeights[attr] = currentWeights[attr] - m_currEta*(gradients[attr] - regularizerComponents[attr]); 

      if (newWeights[attr] <= 0) {
	System.out.println("Prevented 0/- weight " + ((float)newWeights[attr]) 
			   + " for attribute " + m_instances.attribute(attr).name()
			   + ";\tprev=" + ((float)currentWeights[attr])
			   + ";\tgrad=" + ((float)gradients[attr])
			   + ";\treg=" + ((float)regularizerComponents[attr])); 
	newWeights[attr] = m_minWeightValue;
      }
    }
    System.out.print("eta=" + (float)m_currEta); 
    m_currEta = m_currEta * m_etaDecayRate;
    System.out.print(" -> " + (float)m_currEta); 

    // PRINT top weights
    TreeMap map = new TreeMap(Collections.reverseOrder());
    for (int j = 0; j < newWeights.length; j++) {
      map.put(new Double(newWeights[j]), new Integer(j));
    }
    Iterator it = map.entrySet().iterator();
    for (int j=0; j < 5 && it.hasNext(); j++) {
      Map.Entry entry = (Map.Entry) it.next();
      int idx = ((Integer)entry.getValue()).intValue();
      System.out.println("\t" + m_instances.attribute(idx).name() 
			 + "\t" + (float)currentWeights[idx] + "->" + (float)newWeights[idx]
			 + "\tgradient=" + (float)gradients[idx]
			 + "\tregularizer=" + (float)regularizerComponents[idx]);
    }
    // end PRINT top weights
    
    return newWeights; 
  } 
    

  

}
