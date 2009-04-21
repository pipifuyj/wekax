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
 *    Regularizer.java
 *    Copyright (C) 2004 Mikhail Bilenko and Sugato Basu
 *
 */

package weka.clusterers.regularizers; 
import java.io.Serializable;

/** 
 * A parent class for distortion measure regularizers
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu) and Sugato Basu
 * (sugato@cs.utexas.edu)
 * @version $Revision: 1.1 $ */

public abstract class Regularizer implements Cloneable, Serializable {

  /** Compute the regularizer value for given weights */
  public abstract double computeRegularizer(double[] weights);

  /** Compute the gradient of regularizer wrt to given weight */
  public abstract double gradient(double weight); 
  
  /** Create a copy of this regularizer */
  public Object clone() {
    Regularizer reg = null; 
    try {
      reg = (Regularizer) super.clone();
    } catch (CloneNotSupportedException e) {
      System.err.println("Regularizer can't clone");
    }
    // copy the fields below
    
    return reg;
  } 
}



