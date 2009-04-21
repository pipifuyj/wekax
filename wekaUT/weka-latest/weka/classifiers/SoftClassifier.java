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
 *    SoftClassifier.java
 *    Copyright (C) 2003 Ray Mooney
 *
 */

package weka.classifiers;
import weka.core.*;

/** 
 * Interface to a classifier that supports soft classified training data
 * that are SoftClassifiedInstances that have probabilistic class labels
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
 */
public interface SoftClassifier {
    

    /**
   * Generates a classifier. Must initialize all fields of the classifier
   * that are not being set via options (ie. multiple calls of buildClassifier
   * must always lead to the same result). Must not change the dataset
   * in any way.
   *
   * @param data set of instances serving as training data 
   * @exception Exception if the classifier has not been 
   * generated successfully
   */
    public void buildClassifier(SoftClassifiedInstances data) throws Exception;

    /**
   * Predicts the class memberships for a given instance. If
   * an instance is unclassified, the returned array elements
   * must be all zero. If the class is numeric, the array
   * must consist of only one element, which contains the
   * predicted value.
   *
   * @param instance the instance to be classified
   * @return an array containing the estimated membership 
   * probabilities of the test instance in each class (this 
   * should sum to at most 1)
   * @exception Exception if distribution could not be 
   * computed successfully
   */
    public double[] unNormalizedDistributionForInstance(Instance instance) 
	throws Exception;

}


