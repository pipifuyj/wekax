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
 *    SoftClassifiedInstance.java
 *    Copyright (C) 2004 Ray Mooney
 *
 */

package weka.core;

/**
 * An Instance that has a probability distribution across class values.
 * Particularly useful for EM using a SoftClassifier.  Defined as
 * an Interface to allow SoftClassifiedFullInstance and SoftClassifiedSparseInstance
 * to extend Instance and SparseInstance respectively while still respecting
 * the capabilities of a SoftClassifiedInstance
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
*/
public interface SoftClassifiedInstance {

    /** Return the probability the instance is in the given class */
    public double getClassProbability (int classNum);

    /** Set the probability the instance is in the given class */
    public void setClassProbability (int classNum, double prob);

    /** Get the class distribution for this instance */
    public double[] getClassDistribution ();

    /** Set the class distribution for this instance */
    public void setClassDistribution (double[] dist) throws Exception;

}
