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
 *    Rayleigh.java
 *    Copyright (C) 2004 Mikhail Bilenko and Sugato Basu
 *
 */

package weka.clusterers.regularizers; 

import java.io.Serializable;
import java.util.*;

import weka.core.OptionHandler; 

/** 
 * Rayleigh-prior regularization of metric weights:
 * p(w) = w*exp(-w^2/2s^2) / s^2
 * log p(W) = 1/s^2 * sum (log w - w^2 / 2s^2)
 * d/dw (log(p(W))) = 1/s^2 * (1/w - w/s^2)
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu)
 * @version $Revision: 1.2 $ */

public class Rayleigh extends Regularizer implements OptionHandler {

  protected double m_s = 1.0;
  public void setS(double s) { m_s = s; }
  public double getS() { return m_s; } 

  /** Compute the regularizer value for given weights */
  public double computeRegularizer(double[] weights) {
    double val = 0;
    double twoSSquared = 2 * m_s * m_s; 
    
    for (int i = 0; i < weights.length; i++) {
      if (weights[i] >= 0) { 
	val += Math.log(weights[i]) - weights[i] * weights[i] / twoSSquared; 
      } else {
	System.err.println("Zero/negative weight " + i + ": " + weights[i]); 
	val = -Double.MAX_VALUE;
	break;
      } 
    }
    val = val - weights.length * 2 * Math.log(m_s); 

    return val;
  }

  /** Compute the gradient of regularizer wrt to given weight */
  public double gradient(double weight) {
    if (weight != 0) { 
      return (1.0/weight - weight/(m_s * m_s)); 
    } else {
      return 0;
    }
  }

  /** OptionHandler functions */
  public String [] getOptions() {
    String [] options = new String [2];
    int current = 0;

    options[current++] = "-s";
    options[current++] = "" + m_s;

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  public void setOptions(String[] options) throws Exception {
    // TODO: add later 
  }

  public Enumeration listOptions() {
    // TODO: add later 
    return null;
  }
}



