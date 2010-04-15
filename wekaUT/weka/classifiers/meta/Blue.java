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
 *    Blue.java
 *    Copyright (C) 2005 Prem Melville
 *
 */

////////////////////////////////
//
// WARNING: UNDER DEVELOPMENT
//
////////////////////////////////

package weka.classifiers.meta;

import weka.classifiers.*;
import weka.classifiers.bayes.*;
import java.util.*;
import weka.core.*;
import weka.estimators.*;

/**
 * Budgeted Learning by Utility Estimation (BLUE). 
 *
 * Exhaustively estimates the utility of acquiring each feature.
 *
 * Valid options are:<p>
 *
 * -W classname <br>
 * Specify the full class name of a weak classifier as the basis for 
 * bagging (required).<p>
 *
 * @author Prem Melville (melville@cs.utexas.edu) */
public class Blue extends DistributionClassifier
    implements OptionHandler, BudgetedLearner{
    
    /** Use Naive Bayes estimates for feature-value distributions. */
    protected boolean m_UseNaiveBayes = false;
    
    /** Sample queries probabilistically */
    protected boolean m_UseWeightedSampling = false;
    
    /** The attribute estimators. */
    protected Estimator [][] m_Distributions;
    
    /** Costs of acquiring each feature */
    protected double []m_FeatureCosts; 
    
    /** The model base classifier to use */
    protected DistributionClassifier m_Classifier = new weka.classifiers.trees.j48.J48();  
    
    /** Possible selected policies */
    public static final int EXPECTED_UTILITY = 0,
	ROUND_ROBIN = 1,
	DEFAULT_RR = 2,
	ERROR_SAMPLING = 3,
	HBL = 4,
	ERROR_SAMPLING_RR = 5,
	HBL_RR = 6,
	RANDOM = 7,
	EXPECTED_UTILITY_ENTROPY = 8,
	HBL_ENTROPY = 9,
	UNCERTAINTY_SAMPLING = 10,
	CHEAPEST = 11;
    
    public static final Tag[] TAGS_POLICY = { new Tag(EXPECTED_UTILITY, "Expected Utility"),
					      new Tag(ROUND_ROBIN, "Round Robin"),
					      new Tag(DEFAULT_RR, "EU-RR"),
					      new Tag(ERROR_SAMPLING, "Error Sampling"),
					      new Tag(HBL, "Hierarchical BL"),
					      new Tag(ERROR_SAMPLING_RR, "Error Sampling + RR"),
					      new Tag(HBL_RR, "Hierarchical BL + RR"),
					      new Tag(RANDOM, "Random"),
					      new Tag(EXPECTED_UTILITY_ENTROPY, "Expected Utility (Entropy)"),
					      new Tag(HBL_ENTROPY, "Hierarchical BL (Entropy)"),
					      new Tag(UNCERTAINTY_SAMPLING, "Uncertainty Sampling"),
					      new Tag(CHEAPEST, "Cheapest model")};
    
    /** POLICY for feature selection */
    protected int m_Policy = EXPECTED_UTILITY;
    
    /** Possible cheap policies for the first level of HBL */
    public static final int HBL_ERROR_SAMPLING = 0,
	HBL_UNCERTAINTY_SAMPLING = 1,
	HBL_RANDOM = 2;
    
    public static final Tag[] TAGS_HBL = { new Tag(HBL_ERROR_SAMPLING, "Error Sampling"),
					   new Tag(HBL_UNCERTAINTY_SAMPLING, "Uncertainty Sampling"),
					   new Tag(HBL_RANDOM, "Random")};
    
    protected int m_HBLPolicy = HBL_ERROR_SAMPLING;
    
        
    /** Multiplicative factor for HBL - determines how many queries to select using error sampling*/
    protected double m_Alpha = 10;
    
    /** Set to true to turn on debug output */
    protected boolean m_Debug = true;
    
    /** Random number seed */
    protected int m_Seed = 0;
    
    /** Random number generator */
    protected Random m_Random = new Random(m_Seed);
    
    
  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -W classname <br>
   * Specify the full class name of a weak classifier as the basis for 
   * Blue (required).<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
      setUseNaiveBayes(Utils.getFlag('N',options));
      setUseWeightedSampling(Utils.getFlag('S',options));
      
      String policy = Utils.getOption('P', options);
      if (policy.length() != 0) {
	    setPolicy(Integer.parseInt(policy));
	} else {
	    setPolicy(m_Policy);
	}
      
      String hbl_policy = Utils.getOption('H', options);
      if (hbl_policy.length() != 0) {
	    setHBLPolicy(Integer.parseInt(hbl_policy));
	} else {
	    setHBLPolicy(m_HBLPolicy);
	}

      String alpha = Utils.getOption('A', options);
      if (alpha.length() != 0) {
	  setAlpha(Double.parseDouble(alpha));
      } else {
	  setAlpha(m_Alpha);
      }
      String classifierName = Utils.getOption('W', options);
    if (classifierName.length() == 0) {
      throw new Exception("A classifier must be specified with"
			  + " the -W option.");
    }
    setClassifier((DistributionClassifier) (Classifier.forName(classifierName,Utils.partitionOptions(options))));
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
    String [] options = new String [classifierOptions.length + 11];
    int current = 0;
    
    if(getUseNaiveBayes()){
	options[current++] = "-N";
    }
    
    if(getUseWeightedSampling()){
	options[current++] = "-S";
    }

    options[current++] = "-P"; options[current++] = "" + getPolicy().getSelectedTag().getID();

    options[current++] = "-H"; options[current++] = "" + getHBLPolicy().getSelectedTag().getID();

    options[current++] = "-A"; options[current++] = "" + getAlpha();
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
     * Get the value of m_UseNaiveBayes.
     * @return value of m_UseNaiveBayes.
     */
    public boolean getUseNaiveBayes() {
	return m_UseNaiveBayes;
    }
    
    /**
     * Set the value of m_UseNaiveBayes.
     * @param v  Value to assign to m_UseNaiveBayes.
     */
    public void setUseNaiveBayes(boolean  v) {
	m_UseNaiveBayes = v;
    }
    
    /**
     * Get the value of m_UseWeightedSampling.
     * @return value of m_UseWeightedSampling.
     */
    public boolean getUseWeightedSampling() {
	return m_UseWeightedSampling;
    }
    
    /**
     * Set the value of m_UseWeightedSampling.
     * @param v  Value to assign to m_UseWeightedSampling.
     */
    public void setUseWeightedSampling(boolean  v) {
	m_UseWeightedSampling = v;
    }
      
    
    /**
     * Get the value of m_Alpha.
     * @return value of m_Alpha.
     */
    public double getAlpha() {
	return m_Alpha;
    }
    
    /**
     * Set the value of m_Alpha.
     * @param v  Value to assign to m_Alpha.
     */
    public void setAlpha(double  v) {
	this.m_Alpha = v;
    }
    
    
    /**
     * Set the value of m_Policy.
     * @param v  Value to assign to m_Policy.
     */
    public void setPolicy(SelectedTag  v) {
	this.m_Policy = v.getSelectedTag().getID();
    }
  
    /**
     * Get the value of m_Policy.
     * @return value of m_Policy.
     */
    public SelectedTag getPolicy() {
	return new SelectedTag(m_Policy, TAGS_POLICY);
    }
    
    /**
     * Set the value of m_Policy.
     * @param v  Value to assign to m_Policy.
     */
    public void setPolicy(int  v) {
	this.m_Policy = v;
    }  
 

    /**
     * Set the value of m_HBLPolicy.
     * @param v  Value to assign to m_HBLPolicy.
     */
    public void setHBLPolicy(SelectedTag  v) {
	this.m_HBLPolicy = v.getSelectedTag().getID();
    }
  
    /**
     * Get the value of m_HBLPolicy.
     * @return value of m_HBLPolicy.
     */
    public SelectedTag getHBLPolicy() {
	return new SelectedTag(m_HBLPolicy, TAGS_HBL);
    }
    
    /**
     * Set the value of m_HBLPolicy.
     * @param v  Value to assign to m_HBLPolicy.
     */
    public void setHBLPolicy(int  v) {
	this.m_HBLPolicy = v;
    } 

 /**
   * Set the classifier for bagging. 
   *
   * @param newClassifier the Classifier to use.
   */
  public void setClassifier(DistributionClassifier newClassifier) {
      m_Classifier = newClassifier;
  }

  /**
   * Get the classifier used as the classifier
   *
   * @return the classifier used as the classifier
   */
  public DistributionClassifier getClassifier() {
      return m_Classifier;
  }
    
    //Set costs of acquiring each feature 
    public void setFeatureCosts(double []featureCosts){
	m_FeatureCosts = featureCosts; 
    }
    
    /** 
     * Given a set of incomplete instances, select a specified number of instance-feature queries.
     * @param train set of incomplete instances
     * @param num number of instance-feature pairs to selcted for acquiring remaining features
     * @param queryMatrix matrix to track available queries
     * @exception Exception if selection fails
     */
    public Pair []selectInstancesForFeatures(Instances train, int num, boolean [][]queryMatrix) throws Exception{
	Pair []queries = null;
	switch(m_Policy){
	case ROUND_ROBIN:
	    System.out.println("<<Round Robin>>");
	    queries = roundRobin(train, num, queryMatrix);
	    break;
	case EXPECTED_UTILITY:
	    System.out.println("<<Expected Utility>>");
	    queries = expectedUtility(train, num, queryMatrix);
	    break;
	case EXPECTED_UTILITY_ENTROPY:
	    System.out.println("<<Expected Utility using Entropy>>");
	    queries = expectedUtility(train, num, queryMatrix);
	    break;
	case DEFAULT_RR:
	    System.out.println("<<EU + RR>>");
	    queries = expectedUtility(train, num, queryMatrix);
	    break;    
	case ERROR_SAMPLING:
	    System.out.println("<<Error Sampling>>");
	    queries = errorSampling(train, num, queryMatrix);
	    break;
	case UNCERTAINTY_SAMPLING:
	    System.out.println("<<Uncertainty Sampling>>");
	    queries = errorSampling(train, num, queryMatrix);
	    break;    
	case ERROR_SAMPLING_RR:
	    System.out.println("<<Error Sampling + Round Robin>>");
	    queries = errorSampling(train, num, queryMatrix);
	    break;
	case HBL:
	    System.out.println("<<HBL>>");
	    queries = hbl(train, num, queryMatrix);
	    break;
	case HBL_RR:
	    System.out.println("<<HBL + Round Robin>>");
	    queries = hbl(train, num, queryMatrix);
	    break;
	case HBL_ENTROPY:
	    System.out.println("<<HBL + Entropy>>");
	    queries = hbl(train, num, queryMatrix);
	    break;
	case RANDOM:
	    System.out.println("<<Random Sampling>>");
	    queries = randomSampling(train, num, queryMatrix);
	    break; 
	case CHEAPEST:
	    System.out.println("<<Cheapest>>");
	    queries = cheapest(train, num, queryMatrix);
	    break;  
	default:
	    System.err.println("BLUE: Unrecognized selection policy.");
	}
	return queries;
    }
    

    /**
     * Hierarchical Budgeted Learning
     */
    protected Pair []hbl(Instances train, int num, boolean [][]queryMatrix)throws Exception{
	int subsetSize;//size of the subset of queries selected by errorSampling
	if(m_Alpha < 1.0) subsetSize = num;
	else subsetSize = (int) (num * m_Alpha);
	
	ArrayList subList;
	if(subsetSize >= numQueriesAvailable(queryMatrix))
	    subList = generateAllQueries(queryMatrix);//include all queries
	else {
	    Pair []subset=null;
	    switch(m_HBLPolicy){
	    case HBL_ERROR_SAMPLING:
		subset = errorSampling(train, subsetSize, queryMatrix); 
		break;
	    case HBL_UNCERTAINTY_SAMPLING:
		subset = errorSampling(train, subsetSize, queryMatrix); 
		break;
	    case HBL_RANDOM:
		subset = randomSampling(train, subsetSize, queryMatrix); 
		break;
	    default:
		System.err.println("BLUE: Unrecognized HBL policy.");
	    }
	    
	    subList = new ArrayList();
	    for(int i=0; i<subset.length; i++)
		subList.add(subset[i]);
	}
	boolean []featuresAvailable = findAvailableFeatures(subList, train.numAttributes()-1);
	return selectFromAvailable(train, num, subList, featuresAvailable);
    }
    
    
    //Determine which features (columns) have missing values
    protected boolean []findAvailableFeatures(ArrayList allQueries, int numFeatures){
	boolean []featuresAvailable = new boolean[numFeatures];
	Pair curr;
	for(int i=0; i<allQueries.size(); i++){
	    curr = (Pair) allQueries.get(i);
	    featuresAvailable[(int)curr.second] = true;
	}
	return featuresAvailable;
    }
    
    //Count the number of queries available
    protected int numQueriesAvailable(boolean [][]queryMatrix){
	int ctr = 0;
	for(int i=0; i<queryMatrix.length; i++)
	    for(int j=0; j<(queryMatrix[0].length); j++)
		if(!queryMatrix[i][j]) ctr++;
	return ctr;
    }
    
    //Generate the list of all query pairs
    protected ArrayList generateAllQueries(boolean [][]queryMatrix){
	ArrayList allQueries = new ArrayList();
	for(int i=0; i<queryMatrix.length; i++)
	    for(int j=0; j<(queryMatrix[0].length); j++)
		if(!queryMatrix[i][j]) allQueries.add(new Pair(i,j)); 
	return allQueries;
    }
    
    //Select instances using error sampling, then select features for these instances
    protected Pair []errorSampling(Instances train, int num, boolean [][]queryMatrix)throws Exception{
	//Create list of incomplete instances in the training set
	//Score each incomplete instance based on the error sampling score
	//Associate the same score for each query available for the instance
	//Sort queries based on the score
	
	/* Quite often instances will have the same score, in which
	 * case we would like to treat all features from these
	 * instances and equally valuable for selection.  */
	
	if(m_Policy==UNCERTAINTY_SAMPLING || 
	   (m_Policy==HBL && m_HBLPolicy==HBL_UNCERTAINTY_SAMPLING))
	    System.out.println("UNCERTAINTY SAMPLING...");
	else 
	    System.out.println("ERROR SAMPLING...");
	
	//Make a list of pairs of indices of instances in the query matrix and the corresponding score
	int numInstances = train.numInstances();
	int numFeatures = train.numAttributes()-1;
	//create a list of query pairs
	ArrayList allQueries = new ArrayList();
	ArrayList pairList = new ArrayList(); //list of query-score pairs
	double score;
	int numQueries = 0;
	for(int i=0; i<numInstances; i++){
	    int ctr=0;
	    for(int j=0; j<numFeatures; j++)
		if(!queryMatrix[i][j]){
		    allQueries.add(new Pair(i,j)); 
		    ctr++;//counts features available for current instance
		}
	    if(ctr>0){//the instance is incomplete
		//perform error sampling by default
		if(m_Policy==UNCERTAINTY_SAMPLING || 
		   (m_Policy==HBL && m_HBLPolicy==HBL_UNCERTAINTY_SAMPLING))
		    score = -1*calculateMargin(train.instance(i));
		else
		    score = -1*calculateRandomHybridScore(train.instance(i));
		
		//associate score with all available feature queries for this instance
		//the scores are negated only for consistency of ordering
		Pair curr;
		for(int k=numQueries;k<numQueries+ctr;k++){
		    curr = new Pair(k, score);
		    pairList.add(curr);
		}
	    }
	    numQueries += ctr;
	}
	
	assert (numQueries==allQueries.size()) : "Checksum error";
	
	if(m_Policy != ERROR_SAMPLING_RR && m_Policy != HBL_RR )
	    Collections.shuffle(pairList, m_Random);//shuffle so that ties are broken randomly
	//else select all features from one incomplete instance before
	//proceeding to the next

	//sort in DEScending order
	Collections.sort(pairList, new Comparator() {
                public int compare(Object o1, Object o2) {
		    double diff = ((Pair)o1).second - ((Pair)o2).second; 
		    return(diff < 0 ? 1 : diff > 0 ? -1 : 0);
		}
            });
	
	Pair []queries = new Pair[num];
	if(m_Debug) System.out.println("Sorted list:");
	for(int j=0; j<num; j++){
	    if(m_Debug) System.out.println("\t"+((Pair) pairList.get(j)).second+"\t"+((Pair) pairList.get(j)).first);
	    queries[j] = (Pair) allQueries.get((int) ((Pair) pairList.get(j)).first);
	}
	return queries;
    }
    

    
    
    //Select features using a round robin policy
    protected Pair []roundRobin(Instances train, int num, boolean [][]queryMatrix){
	int numInstances = train.numInstances();
	int numFeatures = train.numAttributes()-1;
	//create a list of query pairs
	Pair []queries = new Pair[num];
	int c=0;
	for(int i=0; i<numInstances && c<num; i++)
	    for(int j=0; j<numFeatures && c<num; j++)
		if(!queryMatrix[i][j])
		    queries[c++] = new Pair(i,j); 
	return queries;
    }
    
    //Randomly select num queries
    protected Pair []randomSampling(Instances train, int num, boolean [][]queryMatrix) throws Exception{
	int numInstances = train.numInstances();
	int numFeatures = train.numAttributes()-1;
	//create a list of query pairs
	ArrayList allQueries = new ArrayList();
	for(int i=0; i<numInstances; i++)
	    for(int j=0; j<numFeatures; j++)
		if(!queryMatrix[i][j]) allQueries.add(new Pair(i,j)); 
	
	Collections.shuffle(allQueries, m_Random);
	Pair []queries = new Pair[num];
	for(int i=0; i<num; i++)
	    queries[i] = (Pair) allQueries.get(i);
	return queries;
    }

    //Acquire features in order of increasing cost
    protected Pair []cheapest(Instances train, int num, boolean [][]queryMatrix) throws Exception{
	int numInstances = train.numInstances();
	int numFeatures = train.numAttributes()-1;
	
	//associate feature indices with costs
	Pair []indexCosts = new Pair[numFeatures];
	for(int i=0;i<numFeatures;i++)
	    indexCosts[i] = new Pair(i,m_FeatureCosts[i]);
	
	//sort in AScending order of costs
	Arrays.sort(indexCosts, new Comparator() {
                public int compare(Object o1, Object o2) {
		    double diff = ((Pair)o2).second - ((Pair)o1).second; 
		    return(diff < 0 ? 1 : diff > 0 ? -1 : 0);
		}
            });
	
	//create a list of query pairs
	Pair []queries = new Pair[num];
	int c=0;
	for(int j=0; j<numFeatures && c<num; j++){
	    int featureIndex = (int) indexCosts[j].first;
	    for(int i=0; i<numInstances && c<num; i++)
		if(!queryMatrix[i][featureIndex])
		    queries[c++] = new Pair(i,featureIndex); 
	}
	
	return queries;
    }
 
    //Selected features based on the maximum expected utility of acquiring the feature-value
    protected Pair[]expectedUtility(Instances train, int num, boolean [][]queryMatrix) throws Exception{
	int numInstances = train.numInstances();
	int numFeatures = train.numAttributes()-1;
	//create a list of query pairs
	ArrayList allQueries = new ArrayList();
	boolean []featureAvailable = new boolean[numFeatures];
	for(int i=0; i<numInstances; i++)
	    for(int j=0; j<numFeatures; j++)
		if(!queryMatrix[i][j]){
		    allQueries.add(new Pair(i,j)); 
		    featureAvailable[j] = true;
		    //keep track which features (columns) are still available
		}
	
	//Shuffle all the queries unless the default is Round Robin
	if(m_Policy!=DEFAULT_RR && m_Policy!=HBL_RR) Collections.shuffle(allQueries, m_Random);
	return selectFromAvailable(train, num, allQueries, featureAvailable);
    }
    
    
    protected Pair[]selectFromAvailable(Instances train, int num, ArrayList allQueries, boolean []featureAvailable)throws Exception{
	int numFeatures = train.numAttributes()-1;
	Pair []queries = new Pair[num];
	//Generate a classifier for each available feature
	//For each instance-feature pair compute a score
	//Sort queries by score
	//Return top num queries
	
	/*************************
	 * We are assuming all features are nominal. But this can be
	 * changed by using a discretizer for numeric features and
	 * then treating them as nominal. This can be done by passing
	 * the training set through a filter.
	 *************************/
	
	double currentMeasure = computeCurrentMeasure(train);//accuracy/entropy on training set
	
	int origClassIndex=-1;
	Classifier []featurePredictors=null;
	    
	if(m_UseNaiveBayes){
	    NaiveBayes nb = new NaiveBayes();
	    nb.buildClassifier(train);
	    m_Distributions = nb.getDistributions();
	}else{
	    origClassIndex = train.classIndex();//backup class index
	    featurePredictors = new Classifier [numFeatures];
	    for(int i=0; i<numFeatures; i++){
		if(featureAvailable[i]){
		    Classifier tmp[] = Classifier.makeCopies(m_Classifier,1);
		    featurePredictors[i] = tmp[0]; 
		    train.setClassIndex(i);//set the feature (column) as the target variable
		    featurePredictors[i].buildClassifier(train);
		}
	    }
	    train.setClassIndex(origClassIndex);//reset class index
	}
	
	double []probs=null;
	Pair []pairs = new Pair[allQueries.size()];
	for(int i=0; i<allQueries.size(); i++){
	    Pair curr = (Pair) allQueries.get(i);
	    Instance instance = train.instance((int)curr.first);
	    int featureIndex = (int)curr.second;
	    
	    if(!m_UseNaiveBayes){
		train.setClassIndex(featureIndex);
		probs = ((DistributionClassifier)featurePredictors[featureIndex]).distributionForInstance(instance);
		train.setClassIndex(origClassIndex);//reset class index
	    }
	    //Try this out with 
	    //1) uniform priors, 
	    //2) probabilities estimated from the training data, 
	    //3) and Laplace smoothing

	    double score = computeUtility(instance, featureIndex, probs, train, currentMeasure); 
	    pairs[i] = new Pair(i,score);
	    //Associate the score with the query
	}
	
	//sort in DEScending order
	Arrays.sort(pairs, new Comparator() {
                public int compare(Object o1, Object o2) {
		    double diff = ((Pair)o1).second - ((Pair)o2).second; 
		    return(diff < 0 ? 1 : diff > 0 ? -1 : 0);
		}
            });
	
	if(m_UseWeightedSampling){//use probabilistic selection of queries
	    pairs = sampleWithWeights(pairs, num);
	}//else select top n queries
	
	if(m_Debug) System.out.println("Selected list:");
	for(int j=0; j<num; j++){
	    if(m_Debug) System.out.println("\t"+pairs[j].second+"\t"+pairs[j].first);
	    queries[j] = (Pair) allQueries.get((int) pairs[j].first);
	}
	
	return queries;
    }
    
    /**
     * Sample from a distribution based on assigned scores.
     * 
     * @param pairs array of pairs for object and score
     * @param num number of objects to select
     **/
    protected Pair []sampleWithWeights(Pair []pairs, int num){
	//convert array of pairs to vector
	int poolSize = pairs.length;
	Vector v = new Vector(poolSize);
	for(int i=0; i<poolSize; i++)
	    v.add(pairs[i]);
	
	return sampleWithWeights(v, num);
    }
    
    /**
     * Sample from a distribution based on assigned scores.
     * 
     * @param v vector of pairs for object and score
     * @param num number of objects to select
     **/
    protected Pair []sampleWithWeights(Vector v, int num){
	int poolSize = v.size();
	//Assumes list is in descending order of scores
	//move range to account for any negative values
	double min = ((Pair) v.get(poolSize - 1)).second;
	Pair curr;
	if(min < 0){
	    for(int i=0; i<poolSize; i++){
		curr = (Pair) v.get(i);
		curr.second = curr.second - min;
	    }
	}
	
	Pair []selected = new Pair[num];
	double sum;
	/* For j=1 to n
	 *     Create a cdf
	 *     Randomly pick instance based on cdf
	 *     Note index and remove element 
	 */
	for(int j=0; j<num; j++){
	    sum = 0;
	    for(int i=0; i<v.size(); i++)
		sum += ((Pair) v.get(i)).second;
	    
	    //normalize
//  	    if (Double.isNaN(sum)) {
//  		for(int i=0; i<v.size(); i++)
//  		    System.err.print(((Pair) v.get(i)).second+" ");
//  		System.err.println();
//  		throw new IllegalArgumentException("Can't normalize array. Sum is NaN. Sum = "+sum);
//  	    }
	    //	    if (sum == 0) {
	    if (sum == 0 || Double.isNaN(sum)) {
		System.err.println("Sum = "+sum+", setting to uniform weights.");
		//set probabilities for uniform selection
		double uniform = 1.0/v.size();
		for(int i=0; i<v.size(); i++)
		    ((Pair) v.get(i)).second = uniform;
		sum = 1.0;
	    }else{
		for(int i=0; i<v.size(); i++)
		    ((Pair) v.get(i)).second = ((Pair) v.get(i)).second/sum;
	    }
	    
	    //create a cdf
	    double []cdf = new double[v.size()];
	    cdf[0] = ((Pair) v.get(0)).second;
	    for(int i=1; i<v.size(); i++)
		cdf[i] = ((Pair) v.get(i)).second + cdf[i-1];
	    
	    double rnd = m_Random.nextDouble();
	    int index = 0;
	    while(index < cdf.length && rnd > cdf[index]){
		index++;
	    }
	    selected[j] = (Pair) v.get(index);
	    v.remove(index);
	}
	assert v.size()+num==poolSize : v.size()+" + "+num+" != "+poolSize+"\n";
	
	return selected;
    }
    
    
//      //randomly shuffle the given list
//      protected void shuffle(ArrayList list){
//  	System.out.println("Doing the shuffle...");
//  	Random random = new Random(m_Seed);
//  	Object obj; int loc;
//  	for (int j = list.size()-1; j > 0; j--){
//  	    //swap objects
//  	    obj = list.get(j);
//  	    loc = random.nextInt(j+1);
//  	    list.set(j,list.get(loc));
//  	    list.set(loc,obj);
//  	}
//      }
    
    
    /**
     * Compute the utility of the instance-feature pair. 
     * Expected accuracy, Acc_{t+1} = Sigma_i (P(Fj) * Acc(M(Fj))
     * Score = (Acc_{t+1} - Acc_{t})/Cost_of_Fj
     * Score can be computed for measures other than accuracy, e.g. entropy.
     *
     * @param instance instance under consideration
     * @param featureIndex feature under consideration
     * @param probs predicted probability of each feature-value for the instance
     * @param train training set over which utility is measured
     * @param currentMeasure the accuracy/entropy of the current model
     */
    protected double computeUtility(Instance instance, int featureIndex, double []probs, Instances train, double currentMeasure) throws Exception{
	//For each feature-value with a non-zero probability generate a classifier
	//Measure accuracy of the classifier
	//Compute score as the expected accuracy of the classifier
	double sum = 0.0;
	int numValues = train.attribute(featureIndex).numValues();
	
	Classifier classifier;
	Evaluation eval;
	double utility; 
	//Assumes that probs is actually a distribution i.e. adds up to 1.0
	for(int i=0; i<numValues; i++){
	    if(probs==null || probs[i]!=0){
		Classifier tmp[] = Classifier.makeCopies(m_Classifier,1);
		classifier = tmp[0];
		instance.setValue(featureIndex, i);
		classifier.buildClassifier(train);//train classifier assuming current value for feature
		instance.setMissing(featureIndex);//reset feature to be missing
		//DEBUG should handle Evaluation(train, costMatrix)
		if(m_Policy==EXPECTED_UTILITY_ENTROPY || m_Policy==HBL_ENTROPY){
		    //if(m_Debug) System.out.println("Using entropy...");
		    //compute the expected entropy
		    eval = new Evaluation(train);
		    eval.evaluateModel(classifier, train);
		    utility = -1 * eval.SFMeanSchemeEntropy();
		}else{
		    //compute expected accuracy
		    utility = computeAccuracy(classifier, train);
		    //accuracy = eval.pctCorrect();
		}
		
		if(m_UseNaiveBayes) {
		    sum += m_Distributions[featureIndex][(int)instance.classValue()].getProbability(i)*
			utility;
		}else{
		    sum += probs[i]*utility;
		}
	    }
	}
	return ((sum - currentMeasure)/m_FeatureCosts[featureIndex]);
    }
	
    
    //Compute current model's accuracy/entropy on training set
    protected double computeCurrentMeasure(Instances train) throws Exception{
	Evaluation eval;
	double measure = 0.0;
	if(m_Policy==EXPECTED_UTILITY_ENTROPY || m_Policy==HBL_ENTROPY){
	    //if(m_Debug) System.out.println("Using entropy...");
	    //compute the current (negative) entropy
	    eval = new Evaluation(train);
	    eval.evaluateModel(m_Classifier, train);
	    measure = -1 * eval.SFMeanSchemeEntropy();
	}else{
	    //compute expected accuracy
	    measure = computeAccuracy(m_Classifier, train);
	}
	return measure;
    }
    
	
    /** 
     * Computes the accuracy in classification on the given data.
     *
     * @param data the instances to be classified
     * @return classification accuracy
     * @exception Exception if error can not be computed successfully
     */
    protected double computeAccuracy(Classifier classifier, Instances data) throws Exception {
	double acc = 0.0;
	int numInstances = data.numInstances();
	Instance curr;
	
	for(int i=0; i<numInstances; i++){
	    curr = data.instance(i);
	    //Check if the instance has been correctly classified
	    if(curr.classValue() == ((int) classifier.classifyInstance(curr))) acc++;
	}
	return (acc/numInstances);
    }

    
    //For debugging purposes
    void printArray(double []array){
	for(int i=0; i<array.length; i++)
	    System.out.print(array[i]+" ");
	System.out.println();
    }
    
  /**
   * Build a classifier based on the selected base learner. 
   *
   * @param data the training data to be used for generating the
   * Blue classifier.
   * @exception Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {
      m_Classifier.buildClassifier(data);
  }
    
  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @exception Exception if distribution can't be computed successfully
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
      return m_Classifier.distributionForInstance(instance);
  }
    

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  public String toString() {
      return m_Classifier.toString();
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector(1);
    newVector.addElement(new Option(
	      "\tFull name of classifier to bag.\n"
	      + "\teg: weka.classifiers.trees.j48.J48",
	      "W", 1, "-W"));
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
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
   
    try {
      System.out.println(Evaluation.
			 evaluateModel(new Blue(), argv));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
}
