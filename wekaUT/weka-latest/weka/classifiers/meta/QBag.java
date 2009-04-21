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
 *    QBag.java
 *    Copyright (C) 1999 Eibe Frank
 *    Modified by Prem Melville
 *
 */

package weka.classifiers.meta;

import weka.classifiers.*;
import weka.classifiers.Classifier;
import weka.classifiers.DistributionClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import java.util.*;

import weka.core.*;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.WeightedInstancesHandler;
import weka.core.Option;
import weka.core.Utils;
import weka.core.UnsupportedAttributeTypeException;

/**
 * This class implements Query-by-Bagging based on Abe and Mamitsuka (ICML 98).
 * Built on class for bagging a classifier. For more information, see<p>
 *
 * Leo Breiman (1996). <i>QBag predictors</i>. Machine
 * Learning, 24(2):123-140. <p>
 *
 * Valid options are:<p>
 *
 * -W classname <br>
 * Specify the full class name of a weak classifier as the basis for 
 * bagging (required).<p>
 *
 * -I num <br>
 * Set the number of bagging iterations (default 10). <p>
 *
 * -S seed <br>
 * Random number seed for resampling (default 1). <p>
 *
 * -P num <br>
 * Size of each bag, as a percentage of the training size (default 100). <p>
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (len@reeltwo.com)
 * @author Prem Melville (melville@cs.utexas.edu)
 * @version $Revision: 1.5 $
 */
public class QBag extends EnsembleClassifier 
  implements ActiveLearner, OptionHandler, WeightedInstancesHandler {
    
    /** Set true to use hard assignment for ensemble member votes */
    protected boolean m_HardVoteAssignment = true;
    
    /** Set to true to get debugging output. */
    protected boolean m_Debug = false;
    
  /** The model base classifier to use */
  protected Classifier m_Classifier = new weka.classifiers.trees.j48.J48();  
  
  /** Array for storing the generated base classifiers. */
  protected Classifier[] m_Classifiers;
  
  /** The number of iterations. */
  protected int m_NumIterations = 15;

  /** The seed for random number generation. */
  protected int m_Seed = 1;

  /** The size of each bag sample, as a percentage of the training size */
  protected int m_BagSizePercent = 100;

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(5);

    newVector.addElement(new Option(
	      "\tNumber of bagging iterations.\n"
	      + "\t(default 10)",
	      "I", 1, "-I <num>"));
    newVector.addElement(new Option(
	      "\tFull name of classifier to bag.\n"
	      + "\teg: weka.classifiers.bayes.NaiveBayes",
	      "W", 1, "-W"));
    newVector.addElement(new Option(
              "\tSeed for random number generator.\n"
              + "\t(default 1)",
              "S", 1, "-S"));
    newVector.addElement(new Option(
              "\tSize of each bag, as a percentage of the\n" 
              + "\ttraining set size. (default 100)",
              "P", 1, "-P"));
    newVector.addElement(new Option(
	      "\tTurn on to use hard vote assignment.",
	      "H", 0, "-H"));

    if ((m_Classifier != null) &&
	(m_Classifier instanceof OptionHandler)) {
      newVector.addElement(new Option(
	     "",
	     "", 0, "\nOptions specific to classifier "
	     + m_Classifier.getClass().getName() + ":"));
      Enumeration enum = ((OptionHandler)m_Classifier).listOptions();
      while (enum.hasMoreElements()) {
	newVector.addElement(enum.nextElement());
      }
    }
    return newVector.elements();
  }


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -W classname <br>
   * Specify the full class name of a weak classifier as the basis for 
   * bagging (required).<p>
   *
   * -I num <br>
   * Set the number of bagging iterations (default 10). <p>
   *
   * -S seed <br>
   * Random number seed for resampling (default 1).<p>
   *
   * -P num <br>
   * Size of each bag, as a percentage of the training size (default 100). <p>
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    setHardVoteAssignment(Utils.getFlag('H', options));

    String bagIterations = Utils.getOption('I', options);
    if (bagIterations.length() != 0) {
      setNumIterations(Integer.parseInt(bagIterations));
    } else {
      setNumIterations(10);
    }

    String seed = Utils.getOption('S', options);
    if (seed.length() != 0) {
      setSeed(Integer.parseInt(seed));
    } else {
      setSeed(1);
    }

    String bagSize = Utils.getOption('P', options);
    if (bagSize.length() != 0) {
      setBagSizePercent(Integer.parseInt(bagSize));
    } else {
      setBagSizePercent(100);
    }

    String classifierName = Utils.getOption('W', options);
    if (classifierName.length() == 0) {
      throw new Exception("A classifier must be specified with"
			  + " the -W option.");
    }
    setClassifier(Classifier.forName(classifierName,
				     Utils.partitionOptions(options)));
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] classifierOptions = new String [0];
    if ((m_Classifier != null) && 
	(m_Classifier instanceof OptionHandler)) {
      classifierOptions = ((OptionHandler)m_Classifier).getOptions();
    }
    String [] options = new String [classifierOptions.length + 10];
    int current = 0;
    options[current++] = "-S"; options[current++] = "" + getSeed();
    options[current++] = "-I"; options[current++] = "" + getNumIterations();
    options[current++] = "-P"; options[current++] = "" + getBagSizePercent();
    if(getHardVoteAssignment()){
	options[current++] = "-H";
    }
    
    if (getClassifier() != null) {
      options[current++] = "-W";
      options[current++] = getClassifier().getClass().getName();
    }
    options[current++] = "--";
    System.arraycopy(classifierOptions, 0, options, current, 
		     classifierOptions.length);
    current += classifierOptions.length;
    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


    /**
     * Get the value of m_HardVoteAssignment.
     * @return value of m_HardVoteAssignment.
     */
    public boolean getHardVoteAssignment() {
	return m_HardVoteAssignment;
    }
    
    /**
     * Set the value of m_HardVoteAssignment.
     * @param v  Value to assign to m_HardVoteAssignment.
     */
    public void setHardVoteAssignment(boolean  v) {
	m_HardVoteAssignment = v;
    }
    

  /**
   * Set the classifier for bagging. 
   *
   * @param newClassifier the Classifier to use.
   */
  public void setClassifier(Classifier newClassifier) {

    m_Classifier = newClassifier;
  }

  /**
   * Get the classifier used as the classifier
   *
   * @return the classifier used as the classifier
   */
  public Classifier getClassifier() {

    return m_Classifier;
  }


  /**
   * Gets the size of each bag, as a percentage of the training set size.
   *
   * @return the bag size, as a percentage.
   */
  public int getBagSizePercent() {

    return m_BagSizePercent;
  }
  
  /**
   * Sets the size of each bag, as a percentage of the training set size.
   *
   * @param newBagSizePercent the bag size, as a percentage.
   */
  public void setBagSizePercent(int newBagSizePercent) {

    m_BagSizePercent = newBagSizePercent;
  }
  
  /**
   * Sets the number of bagging iterations
   */
  public void setNumIterations(int numIterations) {

    m_NumIterations = numIterations;
  }

  /**
   * Gets the number of bagging iterations
   *
   * @return the maximum number of bagging iterations
   */
  public int getNumIterations() {
    
    return m_NumIterations;
  }

  /**
   * Set the seed for random number generation.
   *
   * @param seed the seed 
   */
  public void setSeed(int seed) {

    m_Seed = seed;
  }

  /**
   * Gets the seed for the random number generations
   *
   * @return the seed for the random number generation
   */
  public int getSeed() {
    
    return m_Seed;
  }

  /**
   * QBag method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @exception Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {
      //Initialize measures
      initMeasures();

    if (m_Classifier == null) {
      throw new Exception("A base classifier has not been specified!");
    }
    if (data.checkForStringAttributes()) {
      throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
    }
    m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

    int bagSize = data.numInstances() * m_BagSizePercent / 100;
    Random random = new Random(m_Seed);
    for (int j = 0; j < m_Classifiers.length; j++) {
      Instances bagData = data.resampleWithWeights(random);
      if (bagSize < data.numInstances()) {
	bagData.randomize(random);
	Instances newBagData = new Instances(bagData, 0, bagSize);
	bagData = newBagData;
      }
      m_Classifiers[j].buildClassifier(bagData);
    }

    //initialize ensemble wts to be equal 
    m_EnsembleWts = new double [m_NumIterations];
    for(int j=0; j<m_NumIterations; j++)
	m_EnsembleWts[j] = 1.0;
    computeEnsembleMeasures(data);
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @exception Exception if distribution can't be computed successfully
   */
  public double[] distributionForInstance(Instance instance) throws Exception {

    double [] sums = new double [instance.numClasses()], newProbs; 
    
    for (int i = 0; i < m_NumIterations; i++) {
      if (instance.classAttribute().isNumeric() == true) {
	sums[0] += m_Classifiers[i].classifyInstance(instance);
      } else if ((!m_HardVoteAssignment) && (m_Classifiers[i] instanceof DistributionClassifier)) {
	newProbs = ((DistributionClassifier)m_Classifiers[i]).
	  distributionForInstance(instance);
	for (int j = 0; j < newProbs.length; j++)
	  sums[j] += newProbs[j];
      } else {
	sums[(int)m_Classifiers[i].classifyInstance(instance)]++;
      }
    }
    if (instance.classAttribute().isNumeric() == true) {
      sums[0] /= (double)m_NumIterations;
      return sums;
    } else if (Utils.eq(Utils.sum(sums), 0)) {
      return sums;
    } else {
      Utils.normalize(sums);
      return sums;
    }
  }

    /** 
     * Given a set of unlabeled examples, select a specified number of examples to be labeled.
     * @param unlabeledActivePool pool of unlabeled examples
     * @param num number of examples to selected for labeling
     * @exception Exception if selective sampling fails
     */
    public int [] selectInstances(Instances unlabeledActivePool,int num) throws Exception{
	//Make a list of pairs of indices and the corresponding measure of informativenes of examples
	//Sort this in the order of informativeness and return the list of num indices
	int poolSize = unlabeledActivePool.numInstances();
	Pair []pairs = new Pair[poolSize];
	for(int i=0; i<poolSize; i++){
	    pairs[i] = new Pair(i,calculateMargin(unlabeledActivePool.instance(i)));
	}
	//sort in ascending order
	Arrays.sort(pairs, new Comparator() {
                public int compare(Object o1, Object o2) {
		    double diff = ((Pair)o2).second - ((Pair)o1).second; 
		    return(diff < 0 ? 1 : diff > 0 ? -1 : 0);
		}
            });
	int []selected = new int[num];
	if(m_Debug) System.out.println("Sorted list:");
	for(int j=0; j<num; j++){
	    if(m_Debug) System.out.println("\t"+pairs[j].second+"\t"+pairs[j].first);
	    selected[j] = (int) pairs[j].first;
	}
	return selected;
    }

    
    //=============== BEGIN EDIT melville ===============
    /** Returns class predictions of each ensemble member */
    public double []getEnsemblePredictions(Instance instance) throws Exception{
	double preds[] = new double [m_NumIterations];
	for(int i=0; i<m_NumIterations; i++)
	    preds[i] = m_Classifiers[i].classifyInstance(instance);
	
	return preds;
    }

    /** 
     * Returns vote weights of ensemble members.
     *
     * @return vote weights of ensemble members
     */
    public double []getEnsembleWts(){
	return m_EnsembleWts;
    }
    
    /** Returns size of ensemble */
    public double getEnsembleSize(){
	return m_NumIterations;
    }
    //=============== END EDIT melville ===============

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  public String toString() {
    
    if (m_Classifiers == null) {
      return "QBag: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("All the base classifiers: \n\n");
    for (int i = 0; i < m_Classifiers.length; i++)
      text.append(m_Classifiers[i].toString() + "\n\n");
    
    return text.toString();
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
   
    try {
      System.out.println(Evaluation.
			 evaluateModel(new QBag(), argv));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
}
