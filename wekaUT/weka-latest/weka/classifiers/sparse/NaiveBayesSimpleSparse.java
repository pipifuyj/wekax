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
 *    NaiveBayesSimpleSparse.java
 *    Copyright (C) 1999 Eibe Frank
 *    Adapted for SparseInstance's by Mikhail Bilenko  2002
 *
 */

package weka.classifiers.sparse;

import weka.classifiers.Classifier;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesSimple;
import java.io.*;
import java.util.*;
import weka.core.*;

/**
 * Class for building and using a simple Naive Bayes classifier that is
 * adapted for Sparse Instances assuming attribute values are counts of the
 * presence of a descriptive token (e.g. frequency of a word in text categorization) 
 * and assuming a multinomial model for generation of examples/documents.
 * See:
 *  T. Mitchell, Machine Learning, McGraw Hill, 1997, section 6.9 & 6.10
 *  and/or
 *  Andrew McCallum and Kamal Nigam, "A Comparison of Event Models for Naive Bayes Text
 *    Classification", Papers from the AAAI-98 Workshop on Text Categorization, 1998, 
 *    pp. 41--48
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz), Mikhail Bilenko (mbilenko@cs.utexas.edu),
 *         Ray Mooney (mooney@cs.utexas.edu

*/
public class NaiveBayesSimpleSparse extends DistributionClassifier
implements OptionHandler,  WeightedInstancesHandler {

    /** The prior probabilities of the classes. */
    protected double [] m_priors;

    /** Conditional probabilities of each attribute given each class */
    protected double[][] m_condProbs;

    /** The instances used for training. */
    protected Instances m_instances;

    /** The number of classes */
    protected int m_numClasses;

    /** Attribute index for class attribute */
    protected int m_classIndex;

    /** The total number of features */
    protected int m_numAttributes; 

    /** m parameter for Laplace m estimate, corresponding to size of pseudosample */
    protected double m_m = 1.0;

    /** A debug flag */
    protected boolean m_debug = false;

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
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
	    int classIdx = (int)instance.classValue();
	    if (!instance.classIsMissing()) {
		for (int i = 0; i < instance.numValues(); i++) {
		    int attrIdx = instance.index(i);
		    if (attrIdx == m_classIndex) continue;
		    double value = instance.valueSparse(i);
		    if (Instance.isMissingValue(value))
			throw new NoSupportForMissingValuesException("Sparse instance should not have missing value");
		    // Get the array of counts per value per class
		    double incr = value * instance.weight();
		    m_condProbs[attrIdx][classIdx] += incr;
		    totalCounts[classIdx] += incr;
		}
		m_priors[classIdx] += instance.weight();
		numTrainingInstances += instance.weight();
	    }
	}

	// Compute log probabilities for each attribute
	for (int i = 0; i < m_numAttributes; i++) {
	    if (i == m_classIndex) continue;
	    double[] countArray = m_condProbs[i];
	    for(int j = 0; j < m_numClasses; j++){
		// Laplace smoothing
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

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified - must a SparseInstance
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed or if the instance is not a SparseInstance
     */
    public double[] unNormalizedDistributionForInstance(Instance _instance) throws Exception {
	if (! (_instance instanceof SparseInstance)) {
	    throw new Exception ("NaiveBayesSimpleSparse works only with SparseInstance's!");
	}
	SparseInstance instance = (SparseInstance) _instance;
	
	double [] probs = (double[]) m_priors.clone();

	for (int i = 0; i < instance.numValues(); i++) {
	    int attrIdx = instance.index(i);
	    if (attrIdx == m_classIndex) continue;
	    double value = instance.valueSparse(i);
	    double[] condProb = m_condProbs[attrIdx];
	    for (int j = 0; j < m_numClasses; j++) {
		probs[j] += value * condProb[j];   
	    }
	}
	if (m_debug) {
	    System.out.print("Computed class probabilities:\n[ ");
	    for (int i = 0; i < probs.length; i++) 
		System.out.print(probs[i] + "(" + Math.exp(probs[i]) + ") ");
	    System.out.println("]");
	}
	return probs;
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
	double[] logProbs = unNormalizedDistributionForInstance(instance);
	NaiveBayesSimple.normalizeLogs(logProbs);
	return logProbs;
    }

    /** Get Laplace m parameter that controls amouont of smoothing */
    public double getM () {
	return m_m;
    }

    /** Set Laplace m parameter that controls amouont of smoothing */
    public void setM(double m) {
	m_m = m;
    }

    public String mTipText() {
	return "set amount of smoothing (m in m-estimate)";
    }

    public String globalInfo() {
	return "NaiveBayes for sparse instances (e.g.text) using a multinomial model";
    }

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier as a string.
     */
    public String toString() {

	if (m_instances == null) {
	    return "Sparse Naive Bayes: No model built yet.";
	}
	try {
	    StringBuffer text = new StringBuffer("Sparse Naive Bayes:\n");

	    text.append("Prior class probabilities:\n");
	    for (int i = 0; i < m_priors.length; i++) {
		text.append(Utils.doubleToString(m_priors[i], 10, 8) + "(" + Utils.doubleToString(Math.exp(m_priors[i]), 10, 8) + ")\t");
	    }
		    
	    // Only print out probabilities for each attribute in debug mode
	    if (m_debug) { 
		// Go through all attributes, printing out their conditional probabilities for all classes
		for (int i = 0; i < m_numAttributes; i++) {
		    if (i == m_classIndex) continue;
		    double[] condProb = m_condProbs[i];
		    Attribute attribute = m_instances.attribute(i);
		    text.append("Attribute " + attribute.name() + ": ");
		    text.append("[ ");
		    for (int k = 0; k < m_numClasses; k++) {
			text.append(Utils.doubleToString(condProb[k], 10, 8) + "\t");
		    }
		    text.append(" ]\n");
		    text.append("\n");
		}
	    }
	    return text.toString();
	} catch (Exception e) {
	    e.printStackTrace();
	    return new String("Can't print Sparse Naive Bayes classifier: " + e);
	}
    }

    /**
     * Parses a given list of options. Valid options are:<p>
     *
     * -M num <br>
     * Set amount of Laplace m estimate smoothing (size of pseudo sample)
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
	String mString = Utils.getOption('M', options);
	if (mString.length() != 0) {
	    setM(Double.parseDouble(mString));
	}
    }

    /**
     * Gets the current settings of NaiveBayesSimpleSparse.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String [] getOptions() {
	String[] options = new String [2];
	options[0] = "-M";
	options[1] = "" + getM();
	return options;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

	Vector newVector = new Vector(1);
	newVector.addElement(new Option(
					"\tM: Controls amount of Laplace smoothing " +
					"\t(Default = 1)",
					"M", 1,"-M <value>"));
	return newVector.elements();
    }

    
    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String [] argv) {

	Classifier scheme;

	try {
	    scheme = new NaiveBayesSimpleSparse();
	    System.out.println(Evaluation.evaluateModel(scheme, argv));
	} catch (Exception e) {
	    System.err.println(e.getMessage());
	}
    }
}


