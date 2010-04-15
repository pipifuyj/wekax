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
 *    NaiveBayesSimpleSoft.java
 *    Copyright (C) 2003 Ray Mooney
 *
 */

package weka.classifiers.sparse;

import weka.classifiers.*;
import java.io.*;
import java.util.*;
import weka.core.*;

/** 
 * Version of NaiveBayesSimpleSparse that supports training on SoftClassifiedInstances
 * and WeightedInstances for use with SemiSupEM
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
 */
public class NaiveBayesSimpleSparseSoft extends NaiveBayesSimpleSparse implements SoftClassifier, OptionHandler,
								      WeightedInstancesHandler {

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(SoftClassifiedInstances instances) throws Exception {

	if (instances.checkForStringAttributes()) {
	    throw new UnsupportedAttributeTypeException("Sparse Instances are optimized for non-string attributes!");
	}
	if (instances.checkForNominalAttributes()) {
	    throw new UnsupportedAttributeTypeException("Sparse Instances are optimized for non-nominal attributes!");
	}
	if (instances.classAttribute().isNumeric()) {
	    throw new UnsupportedClassTypeException("Sparse Naive Bayes: Class is numeric!");
	}
	if (m_debug) {
	    System.out.println("Training on " + instances.numInstances() + " instances");
	} 
	m_instances = instances;
	m_classIndex = instances.classIndex();
	m_numClasses = instances.numClasses();
	m_numAttributes = instances.numAttributes();
	int numTrainingInstances = 0;

	// Reserve space
	m_priors = new double[m_numClasses];
	m_condProbs = new double[m_numAttributes][m_numClasses];
	double[] totalCounts = new double[m_numClasses];    // stores total count of all tokens in each category

	// Compute counts and sums
	Enumeration enumInsts = instances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    SparseInstance instance = (SparseInstance) enumInsts.nextElement();
	    for (int i = 0; i < instance.numValues(); i++) {
		int attrIdx = instance.index(i);
		if (attrIdx == m_classIndex) continue;
		double value = instance.valueSparse(i);
		if (Instance.isMissingValue(value))
		    throw new NoSupportForMissingValuesException("Sparse instance should not have missing value");
		for (int classIdx = 0; classIdx < m_numClasses; classIdx++) {
		    // increment counts by attribute value weighted by class probability and instance weight
		    double incr = value * ((SoftClassifiedInstance)instance).getClassProbability(classIdx) 
			* instance.weight();
		    m_condProbs[attrIdx][classIdx] += incr;
		    totalCounts[classIdx] += incr;
		}
	    }
	    for (int classIdx = 0; classIdx < m_numClasses; classIdx++) {
		m_priors[classIdx] += ((SoftClassifiedInstance)instance).getClassProbability(classIdx) 
		    * instance.weight();;
		numTrainingInstances +=  instance.weight();
	    }
	}

	// Compute log probabilities for each attribute
	for (int i = 0; i < m_numAttributes; i++) {
	    if (i == m_classIndex) continue;
	    double[] countArray = m_condProbs[i];
	    for(int j = 0; j < m_numClasses; j++){
		countArray[j] = Math.log((countArray[j] + (m_m / m_numAttributes))/(totalCounts[j]+ m_m));
	    }
	}

    	// Calculate priors
	for (int i = 0; i < m_numClasses; i++) {
	    m_priors[i] = Math.log((m_priors[i] + (m_m / m_numClasses)) / (numTrainingInstances + m_m));
	}

	if (m_debug) {
	    System.out.print("Priors: [");
	    for (int i = 0; i < m_priors.length; i++) 
		System.out.print(m_priors[i] + "(" + Math.exp(m_priors[i]) + ") ");
	    System.out.println("]");
	}
    }

    public String globalInfo() {
	return "SoftClassifier version of NaiveBayesSimpleSparse for use with SemiSupEM";
    }

}

