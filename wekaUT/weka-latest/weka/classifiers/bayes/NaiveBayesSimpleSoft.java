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

package weka.classifiers.bayes;

import weka.classifiers.*;
import java.io.*;
import java.util.*;
import weka.core.*;

/** 
 * Version of NaiveBayesSimple that supports training on SoftClassifiedInstances
 * and WeightedInstances for use with SemiSupEM
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
 */
public class NaiveBayesSimpleSoft extends NaiveBayesSimple implements SoftClassifier, OptionHandler,
								      WeightedInstancesHandler {

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(SoftClassifiedInstances instances) throws Exception {

	int attIndex = 0;
	double sum;
    
	if (instances.checkForStringAttributes()) {
	    throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
	}
	if (instances.classAttribute().isNumeric()) {
	    throw new UnsupportedClassTypeException("Naive Bayes: Class is numeric!");
	}
    
	m_Instances = instances;
    
	// Reserve space
	m_Counts = new double[instances.numClasses()]
	    [instances.numAttributes() - 1][0];
	m_Means = new double[instances.numClasses()]
	    [instances.numAttributes() - 1];
	m_Devs = new double[instances.numClasses()]
	    [instances.numAttributes() - 1];
	m_Priors = new double[instances.numClasses()];
	Enumeration enum = instances.enumerateAttributes();
	while (enum.hasMoreElements()) {
	    Attribute attribute = (Attribute) enum.nextElement();
	    if (attribute.isNominal()) {
		for (int j = 0; j < instances.numClasses(); j++) {
		    m_Counts[j][attIndex] = new double[attribute.numValues()];
		}
	    } else {
		for (int j = 0; j < instances.numClasses(); j++) {
		    m_Counts[j][attIndex] = new double[1];
		}
	    }
	    attIndex++;
	}
    
	// Compute counts and sums taking soft class labels into account
	Enumeration enumInsts = instances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    Enumeration enumAtts = instances.enumerateAttributes();
	    attIndex = 0;
		while (enumAtts.hasMoreElements()) {
		    Attribute attribute = (Attribute) enumAtts.nextElement();
		    for (int classNum = 0; classNum < instances.numClasses(); classNum++) {
			double weightedClassProb = ((SoftClassifiedInstance)instance).getClassProbability(classNum) 
			    * instance.weight();
			if (!instance.isMissing(attribute)) {
			    if (attribute.isNominal()) {
				m_Counts[classNum][attIndex]
				    [(int)instance.value(attribute)] += weightedClassProb;
			    } else {
				m_Means[classNum][attIndex] +=
				    instance.value(attribute) * weightedClassProb;
				m_Counts[classNum][attIndex][0] += weightedClassProb;
				m_Devs[classNum][attIndex] += instance.value(attribute) * 
				    instance.value(attribute) * weightedClassProb;
			    }
			}
		    }
		    attIndex++;
		}
		for (int classNum = 0; classNum < instances.numClasses(); classNum++) {
		    m_Priors[classNum] += ((SoftClassifiedInstance)instance).getClassProbability(classNum) 
			* instance.weight();
		}
	}

	// Compute means, and std deviations across complete datset for use
	// when not sufficient class-specific info
	double[] overallMeans = new double[instances.numAttributes() - 1];
	double[] overallDevs = new double[instances.numAttributes() - 1];
	double[] overallCounts = new double[instances.numAttributes() - 1];
	Enumeration enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	    Attribute attribute = (Attribute) enumAtts.nextElement();
	    if (attribute.isNumeric()) {
		for (int j = 0; j < instances.numClasses(); j++) {
		    overallMeans[attIndex] += m_Means[j][attIndex];
		    overallDevs[attIndex] += m_Devs[j][attIndex];
		    overallCounts[attIndex] += m_Counts[j][attIndex][0];
		}
		if (overallCounts[attIndex] !=0)
		    overallMeans[attIndex] /= overallCounts[attIndex];
		overallDevs[attIndex] =  Math.sqrt(overallDevs[attIndex]/overallCounts[attIndex] -
		    overallMeans[attIndex]*overallMeans[attIndex]);
		if (overallDevs[attIndex] <= m_minStdDev || Double.isNaN(overallDevs[attIndex]))
		    overallDevs[attIndex] = m_minStdDev;
	    }
	    attIndex ++;
    	}

	// Compute conditional probs, means, and std deviations
	enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	    Attribute attribute = (Attribute) enumAtts.nextElement();
	    for (int j = 0; j < instances.numClasses(); j++) {
		if (attribute.isNumeric()) {
		    if (m_Counts[j][attIndex][0] != 0) {
			m_Means[j][attIndex] /= m_Counts[j][attIndex][0];
			m_Devs[j][attIndex] = Math.sqrt(m_Devs[j][attIndex]/m_Counts[j][attIndex][0] -
			    m_Means[j][attIndex] * m_Means[j][attIndex]);
			if (m_Devs[j][attIndex] <= m_minStdDev || Double.isNaN(m_Devs[j][attIndex]))
			    // Back-off to class independent Std dev if no data for class
			    m_Devs[j][attIndex] = overallDevs[attIndex];
		    } else { // Back-off to class independent stats if no data for class
			m_Means[j][attIndex] = overallMeans[attIndex];
			m_Devs[j][attIndex] = overallDevs[attIndex];
		    }
		} else if (attribute.isNominal()) {
		    sum = Utils.sum(m_Counts[j][attIndex]);
		    for (int i = 0; i < attribute.numValues(); i++) {
			m_Counts[j][attIndex][i] = Math.log((m_Counts[j][attIndex][i] + (m_m / (double)attribute.numValues()))
							    / (sum + m_m));
		    }
		}
	    }
	    attIndex++;
	}
    
	// Normalize priors with laplace smoothing
	sum = Utils.sum(m_Priors);
	for (int j = 0; j < instances.numClasses(); j++)
	    m_Priors[j] = Math.log ( (m_Priors[j] + (m_m /(double)instances.numClasses()))
				     / (sum + m_m));
    }


}
