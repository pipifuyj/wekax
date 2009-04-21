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
 *    Prototype.java
 *    Copyright (C) 2004 Raymond J. Mooney
 *
 */

package weka.classifiers.misc;

import weka.classifiers.Classifier;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;
import weka.core.*;
import weka.core.metrics.*;

/**
 * Class for building and using a simple prototype classifier.
 * Computes an average/mean/prototype vector for each class.
 * New examples are classified based on computing distance
 * from the instance feature vector to the closest prototype.
 *
 * For real-valued attributes, standard vector mean and Euclidian distance are used.
 * To handle nominal attributes, the distribution of values for each category
 * are computed (as in naive Bayes) as part of the prototype. The distance along a
 * nominal attribute from an instance with a value V for this attribute to the 
 * prototype for a given class is then: 1- P(V|class)
 *
 * In order to make each attribute contribute equally to the distance, values
 * are normalized to [0,1] by setting NormalizeAttributes, which is set by default
 *
 * Predicted class probabilities to make a DistributionClassifier are
 * assumed to be inversely proportional to the distances from the prototypes
 *
 * Borrows some structure from NaiveBayesSimple
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
 * @version $Revision: 1.1 $ 
 *
*/
public class Prototype extends DistributionClassifier implements  WeightedInstancesHandler, OptionHandler{

    /** All the counts for nominal attributes. */
    protected double [][][] m_Counts;
  
    /** The means for numeric attributes. */
    protected double [][] m_Means;

    /** The range (from min to max) taken on by each of the numeric attributes */
    protected double [] m_Ranges;
    
    /** The instances used for training. */
    protected Instances m_Instances;

    /** If set, Normalize all real attribute values between 0 and 1 so that
     *  each dimension contributes equally to distance */
    protected boolean m_NormalizeAttributes = true;

    /**
     * Returns an enumeration describing the available options.. <p>
     *
     * @return an enumeration of all the available options.
     *
     **/
    public Enumeration listOptions () {
	Vector newVector = new Vector(1);
	newVector.addElement(new Option("\tNormal attribute values to [0,1].", "N", 0, "-N"));
	return  newVector.elements();
    }


    /**
     * Parses a given list of options.
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     *
     **/
    public void setOptions (String[] options)
	throws Exception
    {
	setNormalizeAttributes(Utils.getFlag('N', options));
    }


    /**
     * Gets the current settings.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions () {
	String [] options = new String [1];
	int current = 0;
	if (m_NormalizeAttributes) options[current++] = "-N";
	while (current < options.length) {
	    options[current++] = "";
	}
	return  options;
    }


    public void setNormalizeAttributes (boolean v) {
	m_NormalizeAttributes = v;
    }

    public boolean getNormalizeAttributes () {
	return m_NormalizeAttributes;
    }

    public String normalizeAttributesTipText() {
	return "Scale all real-valued attributes to the range [0,1] to equalize contribution of all attributes.";
    }

    /**
     * Returns a string describing this clusterer
     * @return a description of the evaluator suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
	return "Simple algorithm that computes an average or prototype example for each class "+ 
	    "and then classifies instances based on distance to closest prototype";
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
	if (instances.checkForStringAttributes()) {
	    throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
	}
	if (instances.classAttribute().isNumeric()) {
	    throw new UnsupportedClassTypeException("Only nominal class allowed");
	}
    
	m_Instances = instances;
    
	// Reserve space
	m_Counts = new double[instances.numClasses()][instances.numAttributes() - 1][0];
	m_Means = new double[instances.numClasses()][instances.numAttributes() - 1];
	m_Ranges = new double[instances.numAttributes() - 1];
	double[] maxs = new double[instances.numAttributes() - 1];
	double[] mins = new double[instances.numAttributes()- 1];
	Enumeration enum = instances.enumerateAttributes();
	int attIndex = 0;
    	while (enum.hasMoreElements()) {
	    Attribute attribute = (Attribute) enum.nextElement();
	    maxs[attIndex] = Double.NEGATIVE_INFINITY;
	    mins[attIndex] = Double.POSITIVE_INFINITY;
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
    
	// Compute counts and sums 
	Enumeration enumInsts = instances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    int classNum = (int)instance.classValue();
	    Enumeration enumAtts = instances.enumerateAttributes();
	    attIndex = 0;
	    while (enumAtts.hasMoreElements()) {
		Attribute attribute = (Attribute) enumAtts.nextElement();
		if (!instance.isMissing(attribute)) {
		    if (attribute.isNominal()) {
			m_Counts[classNum][attIndex][(int)instance.value(attribute)] += instance.weight();
		    } else {
			double value = instance.value(attribute);
			m_Means[classNum][attIndex] += value * instance.weight();
			m_Counts[classNum][attIndex][0] += instance.weight();
			if (m_NormalizeAttributes) {
			    if (value < mins[attIndex])
				mins[attIndex] = value;
			    if (value > maxs[attIndex])
				maxs[attIndex] = value;
			}
		    }
		}
		attIndex++;
	    }
	}

	// Compute means across complete datset for use
	// when not sufficient class-specific info
	double[] overallMeans = new double[instances.numAttributes() - 1];
	double[] overallCounts = new double[instances.numAttributes() - 1];
	Enumeration enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	while (enumAtts.hasMoreElements()) {
	    Attribute attribute = (Attribute) enumAtts.nextElement();
	    if (attribute.isNumeric()) {
		for (int j = 0; j < instances.numClasses(); j++) {
		    overallMeans[attIndex] += m_Means[j][attIndex];
		    overallCounts[attIndex] += m_Counts[j][attIndex][0];
		}
		if (overallCounts[attIndex] !=0)
		    overallMeans[attIndex] /= overallCounts[attIndex];
	    }
	    attIndex ++;
    	}

	// Compute conditional probs, means,
	enumAtts = instances.enumerateAttributes();
	attIndex = 0;
	double sum = 0;
	while (enumAtts.hasMoreElements()) {
	    Attribute attribute = (Attribute) enumAtts.nextElement();
	    m_Ranges[attIndex] = maxs[attIndex] - mins[attIndex];
	    for (int j = 0; j < instances.numClasses(); j++) {
		if (attribute.isNumeric()) {
		    if (m_Counts[j][attIndex][0] != 0) {
			m_Means[j][attIndex] /= m_Counts[j][attIndex][0];
		    } else { // Back-off to class independent stats if no data for class
			m_Means[j][attIndex] = overallMeans[attIndex];
		    }
		} else if (attribute.isNominal()) {
		    sum = Utils.sum(m_Counts[j][attIndex]);
		    if (sum != 0) 
			for (int i = 0; i < attribute.numValues(); i++) {
			    m_Counts[j][attIndex][i] = m_Counts[j][attIndex][i] / sum;
			}
		}
	    }
	    attIndex++;
	}
	//	System.out.println(toString());
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
	double[] dists = new double[instance.numClasses()];
	for (int j = 0; j < instance.numClasses(); j++) {
	    Enumeration enumAtts = instance.enumerateAttributes();
	    int attIndex = 0;
	    while (enumAtts.hasMoreElements()) {
		Attribute attribute = (Attribute) enumAtts.nextElement();
		if (!instance.isMissing(attribute)) {
		    if (attribute.isNominal()) {
			// Nominal distance is based on 1 - P(value | class)
			dists[j] += Math.pow((1 - m_Counts[j][attIndex][(int)instance.value(attribute)]),2);
		    } else {
			double diff = (instance.value(attribute) - m_Means[j][attIndex]);
			if (m_NormalizeAttributes) {
			    // Scaling by the range is equivalent to scaling all atrributes to [0,1]
			    // and equalizes the contribution of all attributes
			    if (m_Ranges[attIndex] == 0)
				diff = 1.0;
			    else
				diff = diff / m_Ranges[attIndex];
			}
			dists[j] += Math.pow(diff, 2);
		    }
		}
		attIndex++;
	    }
	    // Use inverse of Euclidian distance from prototype as similarity that is normalized
	    // to a probability distribution
	    dists[j] = Math.sqrt(dists[j]);
	    if (dists[j] == 0.0)
		dists[j] = Double.MAX_VALUE;
	    else
		dists[j] = 1 / dists[j];
	}
	Utils.normalize(dists);
	return dists;
    }
	

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier as a string.
     */
    public String toString() {
	if (m_Instances == null) {
	    return "No model built yet.";
	}
	try {
	    StringBuffer text = new StringBuffer("Prototype Model");
	    int attIndex;
      
	    for (int i = 0; i < m_Instances.numClasses(); i++) {
		text.append("\n\nClass " + m_Instances.classAttribute().value(i) 
			    + "\n\n");
		Enumeration enumAtts = m_Instances.enumerateAttributes();
		attIndex = 0;
		while (enumAtts.hasMoreElements()) {
		    Attribute attribute = (Attribute) enumAtts.nextElement();
		    text.append("Attribute " + attribute.name() + "\n");
		    if (attribute.isNominal()) {
			for (int j = 0; j < attribute.numValues(); j++) {
			    text.append(attribute.value(j) + "\t");
			}
			text.append("\n");
			for (int j = 0; j < attribute.numValues(); j++)
			    text.append(Utils.
					doubleToString(m_Counts[i][attIndex][j], 10, 8)
					+ "\t");
		    } else {
			text.append("Mean: " + Utils.
				    doubleToString(m_Means[i][attIndex], 10, 8) + "\t");
		    }
		    text.append("\n\n");
		    attIndex++;
		}
	    }
      
	    return text.toString();
	} catch (Exception e) {
	    return "Can't print Prototype classifier!";
	}
    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String [] argv) {

	Classifier scheme;

	try {
	    scheme = new Prototype();
	    System.out.println(Evaluation.evaluateModel(scheme, argv));
	} catch (Exception e) {
	    System.err.println(e.getMessage());
	}
    }
}


