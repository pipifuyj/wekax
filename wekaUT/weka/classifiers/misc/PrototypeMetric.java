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
 *    PrototypeMetric.java
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
 * Prototype learner for purely real-valued instances that uses
 * a general weka.core.metrics.Metric.
 * Computes an average/mean/prototype vector for each class.
 * New examples are classified based on computing distance
 * from the instance feature vector to the closest prototype using
 * this Metric.
 *
 * By defaults acts as Rocchio-style classifier that uses cosine similarity
 * Assuming text data arff file is already TFIDF weighted.
 *
 * For example see:
 * Joachims, Thorsten, A Probabilistic Analysis of the Rocchio Algorithm with TFIDF 
 * for Text Categorization. Proceedings of International Conference on Machine Learning 
 * (ICML), 1997.
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
 * @version $Revision: 1.1 $ 
 *
*/
public class PrototypeMetric extends DistributionClassifier implements OptionHandler{

    /** Metric to be used to compare intances to prototype instance */
    protected Metric m_Metric = new WeightedDotP();

    /** Prototype instance for each class */
    protected Instance[] m_Prototypes;
    
    /** The instances used for training. */
    protected Instances m_Instances;

    /**
     * Set the distance metric
     *
     * @param s the metric
     */
    public void setMetric (Metric m) {
	m_Metric = m;
    }

    /**
     * Get the distance metric
     *
     * @returns the distance metric used
     */
    public Metric getMetric () {
	return m_Metric;
    }

    /**
     * Returns an enumeration describing the available options.. <p>
     *
     * @return an enumeration of all the available options.
     *
     **/
    public Enumeration listOptions () {
	Vector newVector = new Vector(1);
	newVector.addElement(new Option(
					"\tUse a specific distance metric. (Default=WeightedDotP)\n",
					"M", 1, "-M"));
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
	String metricString = Utils.getOption('M', options);
	if (metricString.length() != 0) {
	  String[] metricSpec = Utils.splitOptions(metricString);
	  String metricName = metricSpec[0]; 
	  metricSpec[0] = "";
	  System.out.println("Metric name: " + metricName + "\nMetric parameters: " + concatStringArray(metricSpec));
	  setMetric(Metric.forName(metricName, metricSpec));
	}
    }


    /**
     * Gets the current settings.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions () {
	String [] options = new String [50];
	int current = 0;

	options[current++] = "-M";
	options[current++] = Utils.removeSubstring(m_Metric.getClass().getName(), "weka.core.metrics.");
	if (m_Metric instanceof OptionHandler) {
	    String[] metricOptions = ((OptionHandler)m_Metric).getOptions();
	    for (int i = 0; i < metricOptions.length; i++) {
		options[current++] = metricOptions[i];
	    }
	}
	
	while (current < options.length) {
	    options[current++] = "";
	}
	return  options;
    }



    /**
     * Returns a string describing this clusterer
     * @return a description of the evaluator suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
	return "Simple algorithm that computes an average or prototype example for each class "+ 
	    "and then classifies instances based on distance to closest prototype using a given metric";
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
	if (instances.checkForStringAttributes() || instances.checkForNominalAttributes()) {
	    throw new UnsupportedAttributeTypeException("Only handles numeric attributes");
	}
	if (instances.classAttribute().isNumeric()) {
	    throw new UnsupportedClassTypeException("Only nominal class allowed");
	}
	m_Instances = instances;
	// Create initial prototype instance for each class
	m_Prototypes = new Instance[instances.numClasses()];
	Instances[] classPartitions = classPartitionInstances(instances);
	for (int j = 0; j < instances.numClasses(); j++) {
	    m_Prototypes[j] = meanInstance(classPartitions[j]);
	}
	m_Metric.buildMetric(instances);
		//	System.out.println(toString());
    }


    /** Partition instances into a set for each class */
    public Instances[] classPartitionInstances (Instances instances) {
	Instances[] classPartitions = new Instances[instances.numClasses()];
	for (int j = 0; j < instances.numClasses(); j++) {
	    classPartitions[j] = new Instances(instances, instances.numInstances());
	}
	Enumeration enumInsts = instances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    int classNum = (int)instance.classValue();
	    classPartitions[classNum].add(instance);
	}
	return classPartitions;
    }
	
    /** Compute a mean instance for all the instances in a set */ 
    public Instance meanInstance(Instances instances) {
	double [] meanVector;
	if (instances.numInstances() !=0 && instances.firstInstance() instanceof SparseInstance)
	    meanVector = meanVectorSparse(instances);
	else
	    meanVector = meanVectorFull(instances);
	// global centroid is generally dense 
	Instance meanInstance = new Instance(1.0, meanVector);
	meanInstance.setDataset(instances);
	return meanInstance;
    }

    /** Compute mean vector for non-sparse instances using meanOrMode method on Instances */
    protected double[] meanVectorFull (Instances instances) {
	double [] meanVector = new double[m_Instances.numAttributes()];
	for (int j = 0; j < instances.numAttributes(); j++) {
	    meanVector[j] = instances.meanOrMode(j); // uses usual meanOrMode
	}
	return meanVector;
    }

    /** Efficiently compute a mean vector for a set of sparse instances */
    protected double[] meanVectorSparse (Instances instances) {
	int numAttributes = instances.numAttributes();
	double[] meanVector = new double[numAttributes];
	double totalWeight = 0;
	for (int j=0; j<instances.numInstances(); j++) {
	    SparseInstance inst = (SparseInstance) (instances.instance(j));
	    totalWeight += inst.weight();
	    for (int i=0; i<inst.numValues(); i++) {
		int index = inst.index(i);
		meanVector[index]  += inst.weight() * inst.valueSparse(i);
	    }
	}
	for (int k=0; k<numAttributes; k++) {
	    meanVector[k] = meanVector[k] / totalWeight;
	}
	return meanVector;
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
	double[] sim = new double[instance.numClasses()];
	for (int j = 0; j < instance.numClasses(); j++) {
	    sim[j] = m_Metric.similarity(instance, m_Prototypes[j]);
	}
	if (Utils.sum(sim) == 0)
	    // If 0 similarity to all class prototypes just use uniform class distribution 
	    for (int j = 0; j < instance.numClasses(); j++) 
		sim[j] = 1;
	Utils.normalize(sim);
	return sim;
    }
	
    /** A little helper to create a single String from an array of Strings
   * @param strings an array of strings
   * @returns a single concatenated string, separated by commas
   */
    public static String concatStringArray(String[] strings) {
	String result = new String();
	for (int i = 0; i < strings.length; i++) {
	    result = result + "\"" + strings[i] + "\" ";
	}
	return result;
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
      
	    for (int i = 0; i < m_Instances.numClasses(); i++) {
		text.append("\n\nClass " + m_Instances.classAttribute().value(i));
		text.append(m_Prototypes[i]);
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


