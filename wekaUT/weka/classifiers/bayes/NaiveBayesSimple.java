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
 *    NaiveBayesSimple.java
 *    Copyright (C) 1999 Eibe Frank
 *
 */

package weka.classifiers.bayes;

import weka.classifiers.Classifier;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;
import weka.core.*;

/**
 * Class for building and using a simple Naive Bayes classifier.
 * Numeric attributes are modelled by a normal distribution. For more
 * information, see<p>
 *
 * Richard Duda and Peter Hart (1973).<i>Pattern
 * Classification and Scene Analysis</i>. Wiley, New York.

 * @author Eibe Frank (eibe@cs.waikato.ac.nz), Ray Mooney (mooney@cs.utexas.edu)
 * @version $Revision: 1.6 $ 
 *
 * Changes by Ray Mooney to handle min Standard Deviation, back-off to class-independent mean and Std Deviation
 * when there is no class-specific data, calculate with logs of probabilities to avoid underflow, 
 * switch to m-estimate smoothing rather than simple Laplace to avoid over-smoothing, and to handle
 * WeightedInstances
*/
public class NaiveBayesSimple extends DistributionClassifier implements OptionHandler, WeightedInstancesHandler{

    /** All the counts for nominal attributes. */
    protected double [][][] m_Counts;
  
    /** The means for numeric attributes. */
    protected double [][] m_Means;

    /** The standard deviations for numeric attributes. */
    protected double [][] m_Devs;

    /** The prior probabilities of the classes. */
    protected double [] m_Priors;

    /** The instances used for training. */
    protected Instances m_Instances;

    /** Constant for normal distribution. */
    protected static double NORM_CONST = Math.sqrt(2 * Math.PI);

    /** default minimum standard deviation */
    protected double m_minStdDev = 1E-6;

    /** m parameter for Laplace m estimate, corresponding to size of pseudosample */
    protected double m_m = 1.0;

    /**
     * Reset to default options
     */
    protected void resetOptions () {
	m_minStdDev = 1e-6;
	m_m = 1.0;
    }

    /**
     * Returns a string describing this clusterer
     * @return a description of the evaluator suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
	return "Simple Bayesian algorithm assuming conditional independence";
    }

    /**
     * Returns an enumeration describing the available options.. <p>
     *
     * @return an enumeration of all the available options.
     *
     **/
    public Enumeration listOptions () {
	Vector newVector = new Vector(2);
	newVector.addElement(new Option(
					"\tM: Controls amount of Laplace smoothing " +
					"\t(Default = 1)",
					"M", 1,"-M <value>"));
	newVector.addElement(new Option("\tminimum allowable standard deviation "
					+"for normal density computation "
					+"\n\t(default 1e-6)"
					,"D",1,"-D <num>"));
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
	resetOptions();
	String mString = Utils.getOption('M', options);
	if (mString.length() != 0) {
	    setM(Double.parseDouble(mString));
	}
	String optionString = Utils.getOption('D', options);
	if (optionString.length() != 0) {
	    setMinStdDev((new Double(optionString)).doubleValue());
	}
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String minStdDevTipText() {
	return "set minimum allowable standard deviation";
    }

    /**
     * Set the minimum value for standard deviation when calculating
     * normal density. Reducing this value can help prevent arithmetic
     * overflow resulting from multiplying large densities (arising from small
     * standard deviations) when there are many singleton or near singleton
     * values.
     * @param m minimum value for standard deviation
     */
    public void setMinStdDev(double m) {
	m_minStdDev = m;
    }

    /**
     * Get the minimum allowable standard deviation.
     * @return the minumum allowable standard deviation
     */
    public double getMinStdDev() {
	return m_minStdDev;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String mTipText() {
	return "set amount of smoothing (m in m-estimate)";
    }

    /** Get Laplace m parameter that controls amouont of smoothing */
    public double getM () {
	return m_m;
    }

    /** Set Laplace m parameter that controls amouont of smoothing */
    public void setM(double m) {
	m_m = m;
    }


    /**
     * Gets the current settings.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions () {
	
	String [] options = new String [4];
	int current = 0;
	options[current++] = "-M";
	options[current++] = "" + getM();
	options[current++] = "-D";
	options[current++] = ""+getMinStdDev();
	return  options;
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {

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
			m_Counts[classNum][attIndex]
			    [(int)instance.value(attribute)] += instance.weight();
		    } else {
			m_Means[classNum][attIndex] +=
			    instance.value(attribute) * instance.weight();
			m_Counts[classNum][attIndex][0] += instance.weight();
			m_Devs[classNum][attIndex] += instance.value(attribute) * 
			    instance.value(attribute) * instance.weight();
		    }
		}
		attIndex++;
	    }
	    m_Priors[classNum] += instance.weight();
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
	
	//	System.out.println(toString());
    }


    /**
     * Calculates the class membership probabilities for the given test instance.
     * Returns vector of unnormalized logs of probabilities for computational reasons.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed
     */
    public double[] unNormalizedDistributionForInstance(Instance instance) throws Exception {
    
	double [] probs = new double[instance.numClasses()];
	int attIndex;
    
	for (int j = 0; j < instance.numClasses(); j++) {
	    probs[j] = 1;
	    Enumeration enumAtts = instance.enumerateAttributes();
	    attIndex = 0;
	    while (enumAtts.hasMoreElements()) {
		Attribute attribute = (Attribute) enumAtts.nextElement();
		if (!instance.isMissing(attribute)) {
		    if (attribute.isNominal()) {
			probs[j] += m_Counts[j][attIndex][(int)instance.value(attribute)];
		    } else {
			probs[j] += normalDens(instance.value(attribute),
					       m_Means[j][attIndex],
					       m_Devs[j][attIndex]);}
		}
		attIndex++;
	    }
	    probs[j] += m_Priors[j];
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
	normalizeLogs(logProbs);
	return logProbs;
    }


    /** Converts an unormalized vector of logs of probabilities into a normalized
     *   distribution that sums to one */
    public static void normalizeLogs(double[] logProbs) {
	// To avoid underflow problems, first scale logProbs by the maximum before
	// converting out of log space
	double max = logProbs[Utils.maxIndex(logProbs)];
	for (int i = 0; i < logProbs.length; i++) {
	    logProbs[i] = Math.exp(logProbs[i] - max);
	}
	Utils.normalize(logProbs);
    }
	

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier as a string.
     */
    public String toString() {

	if (m_Instances == null) {
	    return "Naive Bayes (simple): No model built yet.";
	}
	try {
	    StringBuffer text = new StringBuffer("Naive Bayes (simple)");
	    int attIndex;
      
	    for (int i = 0; i < m_Instances.numClasses(); i++) {
		text.append("\n\nClass " + m_Instances.classAttribute().value(i) 
			    + ": P(C) = " 
			    + Utils.doubleToString(Math.exp(m_Priors[i]), 10, 8)
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
					doubleToString(Math.exp(m_Counts[i][attIndex][j]), 10, 8)
					+ "\t");
		    } else {
			text.append("Mean: " + Utils.
				    doubleToString(m_Means[i][attIndex], 10, 8) + "\t");
			text.append("Standard Deviation: " 
				    + Utils.doubleToString(m_Devs[i][attIndex], 10, 8));
		    }
		    text.append("\n\n");
		    attIndex++;
		}
	    }
      
	    return text.toString();
	} catch (Exception e) {
	    return "Can't print Naive Bayes classifier!";
	}
    }

    /**
     * Density function of normal distribution returning log of probability
     */
    protected double normalDens(double x, double mean, double stdDev) {
    
	double diff = x - mean;
    
	return Math.log (1 / (NORM_CONST * stdDev)) -
	    (diff * diff / (2 * stdDev * stdDev));
    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String [] argv) {

	Classifier scheme;

	try {
	    scheme = new NaiveBayesSimple();
	    System.out.println(Evaluation.evaluateModel(scheme, argv));
	} catch (Exception e) {
	    System.err.println(e.getMessage());
	}
    }
}


