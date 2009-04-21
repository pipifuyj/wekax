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
 *    LearnableMetric.java
 *    Copyright (C) 2004 Mikhail Bilenko
 *
 */

package weka.core.metrics;

import weka.core.*;
import weka.classifiers.*;


/** 
 * Interface to distance metrics that are learned offline
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.2 $
 */

public abstract class SmoothingMetric extends LearnableMetric {

  // turning smoothing on/off
  protected boolean m_useSmoothing = true;

  // the smoothing parameter and corresponding decay
  protected double m_alpha = 10;
  protected double m_alphaDecayRate = 0.9;
  protected double m_currAlpha; 


  public abstract Instance smoothInstance(Instance instance); 

  /** Switch between using and not using DITC smoothing */
  public void setUseSmoothing(boolean useDITC) {
    m_useSmoothing = useDITC;
  } 

  /** Check whether smoothing is used */
  public boolean getUseSmoothing() {
    return m_useSmoothing;
  }

    /** Set the initial value of the smoothing parameter alpha in DITC smoothing */
  public void setAlpha(double alpha) {
    m_alpha = alpha;
  }
  

  /** Get the initial value of the smoothing parameter alpha in DITC smoothing */
  public double getAlpha() {
    return m_alpha;
  }

  /** Get the current value of the smoothing parameter alpha in DITC smoothing */
  public double getCurrAlpha() {
    return m_currAlpha;
  }

  /** Set the initial value of the smoothing parameter alphaDecayRate in DITC smoothing */
  public void setAlphaDecayRate(double alphaDecayRate) {
    m_alphaDecayRate = alphaDecayRate;
  }
  

  /** Get the initial value of the the decay rate of alpha in DITC smoothing */
  public double getAlphaDecayRate() {
    return m_alphaDecayRate;
  }


  /** Update the current value of alpha by the decay rate */
  public void updateAlpha() {
    m_currAlpha = m_currAlpha * m_alphaDecayRate;
  }   
}
