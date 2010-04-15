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
 *    L1.java
 *    Copyright (C) 2004 Mikhail Bilenko and Sugato Basu
 *
 */

package weka.clusterers.regularizers; 
import java.io.Serializable;

/** 
 * L1-norm regularization of metric weights
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu)
 * @version $Revision: 1.1 $ */

public class L1 extends Regularizer {

  /** Compute the regularizer value for given weights */
  public double computeRegularizer(double[] weights) {
    double val = 0; 
    for (int i = 0; i < weights.length; i++) {
      if (weights[i] != 0) { 
	val -= 1/Math.abs(weights[i]); 
      } else {
	val = -Double.MAX_VALUE;
	break;
      } 
    }

    return val;
  }

  /** Compute the gradient of regularizer wrt to given weight */
  public double gradient(double weight) {
    if (weight != 0) { 
      return -1.0 / (weight * weight);
    } else {
      return 0;
    }
  } 
}



