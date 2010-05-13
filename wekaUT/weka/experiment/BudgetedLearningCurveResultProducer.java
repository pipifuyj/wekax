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
 *    BudgetedLearningCurveResultProducer.java
 *    Copyright (C) 2005 Prem Melville
 *
 */

////////////////////////////////
//
// WARNING: UNDER DEVELOPMENT
//
////////////////////////////////

package weka.experiment;

import java.util.*;
import java.io.*;
import weka.classifiers.*;
import weka.core.*;


/**
 * Does an N-fold cross-validation, but generates a learning curve by
 * also varying the number of training examples. Creates a split that
 * uses increasingly larger fractions of the full training set from
 * the N fold but always using the same N-fold test set for
 * testing. If this is applied to an active learner, then the training
 * examples are selected actively by the learner from the pool of
 * unlabeled examples. If this is not used with an active learner, it
 * should produce the same results as the
 * LearningCurveCrossValidationResultProducer.
 *
 * @author Prem Melville (melville@cs.utexas.edu) 
 */

public class BudgetedLearningCurveResultProducer 
    implements ResultProducer, OptionHandler, AdditionalMeasureProducer {

    /** 
     * The directory used when loading feature cost files on demand, null indicates
     * current directory 
     */
    protected File m_FeatureCostDirectory = new File(System.getProperty("user.dir")+"/featureCosts");
    
    /** Costs for acquiring each feature */ 
    protected double []m_FeatureCosts = null;

    /** The dataset of interest */
    protected Instances m_Instances;

    /** The ResultListener to send results to */
    protected ResultListener m_ResultListener = new CSVResultListener();

    /** The number of folds in the cross-validation */
    protected int m_NumFolds = 10;

    /** Save raw output of split evaluators --- for debugging purposes */
    protected boolean m_debugOutput = false;

    /** The output zipper to use for saving raw splitEvaluator output */
    protected OutputZipper m_ZipDest = null;

    /** The destination output file/directory for raw output */
    protected File m_OutputFile = new File(
					   new File(System.getProperty("user.dir")), 
					   "splitEvalutorOut.zip");

    /** The SplitEvaluator used to generate results */
    //protected SplitEvaluator m_SplitEvaluator = new ClassifierSplitEvaluator();
    protected SplitEvaluator m_SplitEvaluator = new FeatureCostSensitiveClassifierSplitEvaluator();

    /** The names of any additional measures to look for in SplitEvaluators */
    protected String [] m_AdditionalMeasures = null;

    /** 
     * The minimum number of instances to use. If this is zero, the first
     * step will contain m_StepSize instances 
     */
    protected double m_LowerSize = 0;
  
    /**
     * The maximum number of instances to use. -1 indicates no maximum 
     * (other than the total number of instances)
     */
    protected double m_UpperSize = -1;

    /** The number of instances to add at each step */
    protected double m_StepSize = 1;

    /** The specific points to plot, either integers representing specific numbers of training examples,
     * or decimal fractions representing percentages of the full training set*/
    protected double[] m_PlotPoints;

    /** The current dataset size during stepping */
    protected int m_CurrentSize = 0;

    /** Cost to build current model */
    protected double m_Cost = 0.0;
    
    /* The name of the key field containing the dataset name */
    public static String DATASET_FIELD_NAME = "Dataset";

    /* The name of the key field containing the run number */
    public static String RUN_FIELD_NAME = "Run";

    /* The name of the key field containing the fold number */
    public static String FOLD_FIELD_NAME = "Fold";

    /* The name of the result field containing the timestamp */
    public static String TIMESTAMP_FIELD_NAME = "Date_time";
    
    /* The name of the result field containing the sample selection time */
    public static String SELECTION_TIME_FIELD_NAME = "Selection_time";
    
    /* The name of the key field containing the learning rate step number */
    public static String STEP_FIELD_NAME = "Total_instances";

    /* The name of the key field containing the fraction of total instances used */
    public static String FRACTION_FIELD_NAME = "Fraction_instances";
    
    public static String COST_FIELD_NAME = "Model_cost";
    
    /* Extension for feature cost files */
    public static String FEATURE_COST_FILE_EXTENSION = ".featureCosts";
    
    /* Indicates whether fractions or actual number of instances have been specified */
    protected boolean m_IsFraction = false;
    
    /**
     * Returns a string describing this result producer
     * @return a description of the result producer suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
	return "Performs a learning-curve cross validation run using a supplied "
	    +"split evaluator. Trains on increasing subsets of the training data for each split, "
	    +"repeatedly testing on the test set for that split after training on subsets of various sizes.";
    }
    
    

    /**
     * Sets the dataset that results will be obtained for.
     *
     * @param instances a value of type 'Instances'.
     */
    public void setInstances(Instances instances) {
    
	m_Instances = instances;
	loadFeatureCosts();
    }
    
    
    
    /**
     * Returns the directory that will be searched for feature cost files when
     * loading on demand.
     *
     * @return The cost file search directory.
     */
    public File getFeatureCostDirectory() {
	return m_FeatureCostDirectory;
    }
    
    /**
     * Sets the directory that will be searched for feature cost files when
     * loading on demand.
     *
     * @param newDir The cost file search directory.
     */
    public void setFeatureCostDirectory(File newDir) {
	if (newDir.isDirectory()) {
	    m_FeatureCostDirectory = newDir;
	} else {
	    m_FeatureCostDirectory = new File(newDir.getParent());
	}
    }
    
    //Read feature costs from file
    protected void loadFeatureCosts(){
	try{
	    String costName = m_Instances.relationName() + FEATURE_COST_FILE_EXTENSION;
	    File costFile = new File(getFeatureCostDirectory(), costName);
	    if (!costFile.exists()) {
		throw new Exception("Feature cost file doesn't exist: " + costFile);
	    }
	    int numFeatures = numFeatures();
	    m_FeatureCosts = new double[numFeatures];
	    
	    BufferedReader bin = new BufferedReader(new FileReader(costFile));
	    String line = bin.readLine();//read first line - ignore rest of file
	    bin.close();
	    StringTokenizer st = new StringTokenizer(line);
	    int ctr = 0;//count number of costs in the file
	    while(st.hasMoreTokens()) {
		if(ctr < numFeatures){
		    m_FeatureCosts[ctr] = Double.parseDouble(st.nextToken());
		    ctr++;
		}else{
		    //throw new Exception("Feature cost file has too many costs.");   
		    System.err.println("Feature cost file has too many costs: "+costFile);   
		    System.exit(0);
		}
	    }
	    if(ctr < numFeatures){
		System.err.println("Feature cost file has too few costs: "+costFile);   
		System.exit(0);
		//throw new Exception("Feature cost file has too few costs.");   
	    }
	}catch (Exception e){
	    System.err.println (e);
	}
	//DEBUG
	Utils.print(m_FeatureCosts);
    }
    
    /**
     * Sets the object to send results of each run to.
     *
     * @param listener a value of type 'ResultListener'
     */
    public void setResultListener(ResultListener listener) {

	m_ResultListener = listener;
    }

    
    /**
     * Set a list of method names for additional measures to look for
     * in SplitEvaluators. This could contain many measures (of which only a
     * subset may be produceable by the current SplitEvaluator) if an experiment
     * is the type that iterates over a set of properties.
     * @param additionalMeasures an array of measure names, null if none
     */
    public void setAdditionalMeasures(String [] additionalMeasures) {
	m_AdditionalMeasures = additionalMeasures;

	if (m_SplitEvaluator != null) {
	    System.err.println("LearningCurveCrossValidationResultProducer: setting additional "
			       +"measures for "
			       +"split evaluator");
	    m_SplitEvaluator.setAdditionalMeasures(m_AdditionalMeasures);
	}
    }

    /**
     * Returns an enumeration of any additional measure names that might be
     * in the SplitEvaluator
     * @return an enumeration of the measure names
     */
    public Enumeration enumerateMeasures() {
	Vector newVector = new Vector();
	if (m_SplitEvaluator instanceof AdditionalMeasureProducer) {
	    Enumeration en = ((AdditionalMeasureProducer)m_SplitEvaluator).
		enumerateMeasures();
	    while (en.hasMoreElements()) {
		String mname = (String)en.nextElement();
		newVector.addElement(mname);
	    }
	}
	return newVector.elements();
    }
  
    /**
     * Returns the value of the named measure
     * @param measureName the name of the measure to query for its value
     * @return the value of the named measure
     * @exception IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {
	if (m_SplitEvaluator instanceof AdditionalMeasureProducer) {
	    return ((AdditionalMeasureProducer)m_SplitEvaluator).
		getMeasure(additionalMeasureName);
	} else {
	    throw new IllegalArgumentException("LearningCurveCrossValidationResultProducer: "
					       +"Can't return value for : "+additionalMeasureName
					       +". "+m_SplitEvaluator.getClass().getName()+" "
					       +"is not an AdditionalMeasureProducer");
	}
    }
  
    /**
     * Gets a Double representing the current date and time.
     * eg: 1:46pm on 20/5/1999 -> 19990520.1346
     *
     * @return a value of type Double
     */
    public static Double getTimestamp() {

	Calendar now = Calendar.getInstance(TimeZone.getTimeZone("UTC"));
	double timestamp = now.get(Calendar.YEAR) * 10000
	    + (now.get(Calendar.MONTH) + 1) * 100
	    + now.get(Calendar.DAY_OF_MONTH)
	    + now.get(Calendar.HOUR_OF_DAY) / 100.0
	    + now.get(Calendar.MINUTE) / 10000.0;
	return new Double(timestamp);
    }
  
    /**
     * Prepare to generate results.
     *
     * @exception Exception if an error occurs during preprocessing.
     */
    public void preProcess() throws Exception {

	if (m_SplitEvaluator == null) {
	    throw new Exception("No SplitEvalutor set");
	}
	if (m_ResultListener == null) {
	    throw new Exception("No ResultListener set");
	}
	m_ResultListener.preProcess(this);
    }
  
    /**
     * Perform any postprocessing. When this method is called, it indicates
     * that no more requests to generate results for the current experiment
     * will be sent.
     *
     * @exception Exception if an error occurs
     */
    public void postProcess() throws Exception {

	m_ResultListener.postProcess(this);

	if (m_debugOutput) {
	    if (m_ZipDest != null) {
		m_ZipDest.finished();
		m_ZipDest = null;
	    }
	}
    }
  
    /**
     * Gets the keys for a specified run number. Different run
     * numbers correspond to different randomizations of the data. Keys
     * produced should be sent to the current ResultListener
     *
     * @param run the run number to get keys for.
     * @exception Exception if a problem occurs while getting the keys
     */
    public void doRunKeys(int run) throws Exception {
	int numExtraKeys;
	if(m_IsFraction)
	    numExtraKeys = 5;
	else numExtraKeys = 4;
	
	if (m_Instances == null) {
	    throw new Exception("No Instances set");
	}
	if (m_ResultListener == null) {
	    throw new Exception("No ResultListener set");
	}
	for (int fold = 0; fold < m_NumFolds; fold++) {
	    int pointNum = 0;
	    // For each subsample size
	    if (m_PlotPoints != null) {
		m_CurrentSize = plotPoint(0);
	    }
	    else if (m_LowerSize == 0) {
		m_CurrentSize = stepSize(m_StepSize);
	    } else {
		m_CurrentSize = lowerSize(m_LowerSize);
	    }
	    while (m_CurrentSize <= maxNumQueries()) {
		// Add in some fields to the key like run and fold number, dataset name
		Object [] seKey = m_SplitEvaluator.getKey();
		Object [] key = new Object [seKey.length + numExtraKeys];
		key[0] = Utils.backQuoteChars(m_Instances.relationName());
		key[1] = "" + run;
		key[2] = "" + (fold + 1);
		key[3] = "" + m_CurrentSize;
		if(m_IsFraction) key[4] = "" + m_PlotPoints[pointNum];
		System.arraycopy(seKey, 0, key, numExtraKeys, seKey.length);
		if (m_ResultListener.isResultRequired(this, key)) {
		    try {
			m_ResultListener.acceptResult(this, key, null);
		    } catch (Exception ex) {
			// Save the train and test datasets for debugging purposes?
			throw ex;
		    }
		}
		if (m_PlotPoints != null) {
		    pointNum ++;
		    m_CurrentSize = plotPoint(pointNum);
		}
		else {
		    m_CurrentSize += stepSize(m_StepSize);
		}
	    }
	}
    }

    
    /** 
     * Get the maximum number of queries base on the upperSize limit
     * or maximum training set size from the n-fold CV 
     */
    protected int maxNumQueries() {
	if (m_UpperSize == -1 || m_PlotPoints != null)
	    return ((int)(m_Instances.numInstances()*(1 - 1/((double)m_NumFolds))))*numFeatures();
	else if(isInteger(m_UpperSize))
	    return (int) m_UpperSize;
	else return (int) (m_UpperSize*((int)(m_Instances.numInstances()*(1 - 1/((double)m_NumFolds))))*numFeatures());
    }
    
    //If lowersize is a fraction, compute as a fraction of total number of features.
    protected int lowerSize(double lowerSize){
	if(isInteger(lowerSize))
	    return (int) lowerSize;
	else return (int) (lowerSize * ((int)(m_Instances.numInstances()*(1 - 1/((double)m_NumFolds))))*numFeatures());
    }
    
    //If lowersize is a fraction, compute as a fraction of total number of features.
    protected int stepSize(double stepSize){
	int out;
	if(isInteger(stepSize))
	    out = (int) stepSize;
	else out = (int) (stepSize * ((int)(m_Instances.numInstances()*(1 - 1/((double)m_NumFolds))))*numFeatures());
	//step size must at least be 1
	if(out<1) out=1;
	return out;
    }
    
    //Returns the number of features in the datasets (excluding the class attribute)
    protected int numFeatures(){
	return (m_Instances.numAttributes() - 1);
    }
    
    /**
     * Gets the results for a specified run number. Different run
     * numbers correspond to different randomizations of the data. Results
     * produced should be sent to the current ResultListener
     *
     * @param run the run number to get results for.
     * @exception Exception if a problem occurs while getting the results
     */
    public void doRun(int run) throws Exception {
	int numExtraKeys;
	if(m_IsFraction)
	    numExtraKeys = 5;
	else numExtraKeys = 4;
	
	if (getRawOutput()) {
	    if (m_ZipDest == null) {
		m_ZipDest = new OutputZipper(m_OutputFile);
	    }
	}
	if (m_Instances == null) {
	    throw new Exception("No Instances set");
	}
	if (m_ResultListener == null) {
	    throw new Exception("No ResultListener set");
	}
	//Make local copy of data
	//Initialize dataset - based on initial ablation level
	//Run classifier
	//Get queries
	//Update dataset with result of query
	
	if(m_SplitEvaluator instanceof FeatureCostSensitiveClassifierSplitEvaluator){
	    ((FeatureCostSensitiveClassifierSplitEvaluator)m_SplitEvaluator).setFeatureCosts(m_FeatureCosts);
	}
	
	int numFeatures = numFeatures();
	// Randomize on a copy of the original dataset
	Instances runInstances = new Instances(m_Instances);
	runInstances.randomize(new Random(run));
	if (runInstances.classAttribute().isNominal()) {
	    runInstances.stratify(m_NumFolds);
	}

	for (int fold = 0; fold < m_NumFolds; fold++) {//For each fold
	    m_Cost = 0.0;//initialize cost counter
	    Instances fullTrain = runInstances.trainCV(m_NumFolds, fold);
	    // Randomly shuffle stratified training set for fold
	    fullTrain.randomize(new Random(fold));
	    Instances train = initializeData(fullTrain);
	    boolean [][]queryMatrix = new boolean[train.numInstances()][numFeatures];
	    //initially queryMatrix is set to all false

	    boolean firstPoint = true;
	    int prevSize = 0;
	    Instances test = runInstances.testCV(m_NumFolds, fold);
	    int pointNum = 0;
	    // For each subsample size
	    if (m_PlotPoints != null) {
		m_CurrentSize = plotPoint(0);
	    }
	    else if (m_LowerSize == 0) {
		m_CurrentSize = stepSize(m_StepSize);
	    } else {
		m_CurrentSize = lowerSize(m_LowerSize);
	    }
	    
	    //maxQueries should not exceed total number of queries possible in current fold
	    int maxQueries = maxNumQueries();
	    int trainSize = fullTrain.numInstances()*numFeatures();
	    if(maxQueries > trainSize) maxQueries = trainSize;
	    
	    //System.out.println(train);
	    
	    while (m_CurrentSize <= maxQueries) {
		long selectionTimeStart, selectionTimeElapsed;
		// Add in some fields to the key like run and fold number, dataset name
		Object [] seKey = m_SplitEvaluator.getKey();
		Object [] key = new Object [seKey.length + numExtraKeys];
		key[0] = Utils.backQuoteChars(m_Instances.relationName());
		key[1] = "" + run;
		key[2] = "" + (fold + 1);
		key[3] = "" + m_CurrentSize;
		if(m_IsFraction) key[4] = "" + m_PlotPoints[pointNum];
		System.arraycopy(seKey, 0, key, numExtraKeys, seKey.length);
		if (m_ResultListener.isResultRequired(this, key)) {
		    try {
			if(m_IsFraction)
			    System.out.println("Run:" + run + " Fold:" + fold + " Size:" + m_CurrentSize + " Fraction:" + m_PlotPoints[pointNum]);
			else
			    System.out.println("Run:" + run + " Fold:" + fold + " Size:" + m_CurrentSize);
			
			if(firstPoint){//the first training set is always randomly selected
			    firstPoint = false;
			    selectionTimeStart = System.currentTimeMillis();
			    makeRandomQueries(train,fullTrain,m_CurrentSize - prevSize,queryMatrix,run*1000+fold);
			    selectionTimeElapsed = System.currentTimeMillis() - selectionTimeStart;
			}else{
			    //use current classifier to actively select instance-feature pairs 
			    //acquire features-values and add to the training set
			    selectionTimeStart = System.currentTimeMillis(); 
			    makeQueries(train,fullTrain,m_CurrentSize - prevSize,queryMatrix,run*1000+fold);
			    selectionTimeElapsed = System.currentTimeMillis() - selectionTimeStart;
			}
			
			Object [] seResults = m_SplitEvaluator.getResult(train, test);
			Object [] results = new Object [seResults.length + 3];
			results[0] = getTimestamp();
			results[1] = new Double(selectionTimeElapsed / 1000.0);
			results[2] = new Double(m_Cost);
			System.arraycopy(seResults, 0, results, 3,
					 seResults.length);
			if (m_debugOutput) {
			    String resultName = (""+run+"."+(fold+1)+"."+ m_CurrentSize + "." 
						 + Utils.backQuoteChars(runInstances.relationName())
						 +"."
						 +m_SplitEvaluator.toString()).replace(' ','_');
			    resultName = Utils.removeSubstring(resultName, 
							       "weka.classifiers.");
			    resultName = Utils.removeSubstring(resultName, 
							       "weka.filters.");
			    resultName = Utils.removeSubstring(resultName, 
							       "weka.attributeSelection.");
			    m_ZipDest.zipit(m_SplitEvaluator.getRawResultOutput(), resultName);
			}
			m_ResultListener.acceptResult(this, key, results);
		    } catch (Exception ex) {
			// Save the train and test datasets for debugging purposes?
			throw ex;
		    }
		}
		prevSize = m_CurrentSize;
		if (m_PlotPoints != null) {
		    pointNum ++;
		    m_CurrentSize = plotPoint(pointNum);
		}
		else {
		    m_CurrentSize += stepSize(m_StepSize);
		}
		
		//System.out.println(train);
	    }
	}
    }
    
    
    //Create initial training set with only the class labels
    protected Instances initializeData(Instances fulltrain){
	int numInstances = fulltrain.numInstances();
	int numAtts = fulltrain.numAttributes();
	int classIndex = fulltrain.classIndex();
	Instances train = new Instances(fulltrain, numInstances);
	for(int i=0; i<numInstances; i++){
	    Instance newInst = new Instance(numAtts);
	    newInst.setValue(classIndex,(fulltrain.instance(i)).classValue());
	    train.add(newInst);
	}
	return train;
    }
    
    /**
     * Use current classifier to actively select specified number of instances-feature queries
     * to be made. Update training set and the query matrix
     *
     * @param train instances with missing feature-values
     * @param fulltrain instances with all feature-values
     * @param num number of queries to make
     * @param queryMatrix matrix to track available queries
     * @param seed random seed needed if the learner is not a BudgetedLearner 
     */
    protected void makeQueries(Instances train, Instances fulltrain, int num, boolean [][]queryMatrix, int seed) throws Exception{
	Classifier classifier;
	try{
	    classifier = ((ClassifierSplitEvaluator)m_SplitEvaluator).getClassifier();
	}catch (Exception ex){
	    throw new Exception("Budgeted learning is only implemented for evaluators of classifiers.");
	}
	
	if(classifier instanceof BudgetedLearner){
	    //get instance-feature pairs picked by the classifier
	    Pair []queries = ((BudgetedLearner)classifier).selectInstancesForFeatures(train, num, queryMatrix);
	    if(queries.length!=num) throw new Exception("Incorrect number of queries made!");
	    transferQueries(train, fulltrain, queries, queryMatrix);
	}else{//randomly pick examples from local pool
	    makeRandomQueries(train, fulltrain, num, queryMatrix, seed);
	}
    }
    
    //Randomly select queries to make
    protected void makeRandomQueries(Instances train, Instances fulltrain, int num, boolean [][]queryMatrix, int seed) throws Exception{
	int numInstances = fulltrain.numInstances();
	int numFeatures = numFeatures();
	//create a list of query pairs
	ArrayList allQueries = new ArrayList();
	for(int i=0; i<numInstances; i++)
	    for(int j=0; j<numFeatures; j++)
		if(!queryMatrix[i][j]) allQueries.add(new Pair(i,j)); 
	
	//randomly select num queries
	Pair []queries = new Pair[num];
	Random random = new Random(seed);
	System.out.println("Making random queries ...");
	System.out.print(allQueries.size()+" - "+num+" = ");
	for(int i=0; i<num; i++){
	    int index = random.nextInt(allQueries.size());
	    queries[i] = (Pair) allQueries.get(index);
	    allQueries.remove(index);
	}
	System.out.println(allQueries.size());
	transferQueries(train, fulltrain, queries, queryMatrix);
    }
    
    //Transfer requested feature-value to training set 
    protected void transferQueries(Instances train, Instances fulltrain, Pair []queries, boolean [][]queryMatrix){
	int instanceIndex, featureIndex;
	for(int i=0; i<queries.length; i++){
	    instanceIndex = (int) queries[i].first;
	    featureIndex = (int) queries[i].second;
	    if(queryMatrix[instanceIndex][featureIndex]){
		System.err.println("Query tracking failure!");
	    }else{
		(train.instance(instanceIndex)).setValue(featureIndex,(fulltrain.instance(instanceIndex)).value(featureIndex));
		m_Cost += m_FeatureCosts[featureIndex];
		System.out.println("Query for ("+instanceIndex+","+featureIndex+") = "+ (train.instance(instanceIndex)).value(featureIndex) + "\tCost = "+m_FeatureCosts[featureIndex]+"\tTotalCost = "+m_Cost);
		queryMatrix[instanceIndex][featureIndex] = true;
	    }
	}
    }
    
    /**
     * Add new instances to the given set of instances.
     *
     * @param data given instances
     * @param newData set of instances to add to given instances
     */
    protected void addInstances(Instances data, Instances newData){
	for(int i=0; i<newData.numInstances(); i++)
	    data.add(newData.instance(i));
    }
    
    /** Determines if the points specified are fractions of the total number of examples */
    protected boolean setIsFraction(){
	if (m_PlotPoints != null){
	    if(!isInteger(m_PlotPoints[0]))//if the first point is not an integer
		m_IsFraction = true;
	    else
		m_IsFraction = false;
	}//  else{
//  	    if(!isInteger(m_StepSize))//if the step size is not an integer
//  		m_IsFraction = true;
//  	    else
//  		m_IsFraction = false; 
//  	}
	return m_IsFraction;
    }
    
    /** Return the number of instance-feature queries for the ith point on the
     * curve for plotPoints as specified.
     */
    protected int plotPoint(int i) {
	// If i beyond number of given plot points return a value greater than maximum queries
	if (i >= m_PlotPoints.length)
	    return maxNumQueries() + 1;
	double point = m_PlotPoints[i];
	// If plot point is an integer (other than a non-initial 1)
	// treat it as a specific number of queries
	if (isInteger(point) && !(Utils.eq(point, 1.0) && i!=0))
	    return (int)point;
	else
	    // Otherwise, treat it as a percentage of the full set
	    return (int)Math.round(point * maxNumQueries());
    }

    /** Return true if the given double represents an integer value */
    protected static boolean isInteger(double val) {
	return Utils.eq(Math.floor(val), Math.ceil(val));
    }

    /**
     * Gets the names of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @return an array containing the name of each column
     */
    public String [] getKeyNames() {

	String [] keyNames = m_SplitEvaluator.getKeyNames();
	// Add in the names of our extra key fields
	int numExtraKeys;
	if(m_IsFraction)
	    numExtraKeys = 5;
	else numExtraKeys = 4;
	String [] newKeyNames = new String [keyNames.length + numExtraKeys];
	newKeyNames[0] = DATASET_FIELD_NAME;
	newKeyNames[1] = RUN_FIELD_NAME;
	newKeyNames[2] = FOLD_FIELD_NAME;
	newKeyNames[3] = STEP_FIELD_NAME;
	if(m_IsFraction) newKeyNames[4] = FRACTION_FIELD_NAME;
	System.arraycopy(keyNames, 0, newKeyNames, numExtraKeys, keyNames.length);
	return newKeyNames;
    }

    /**
     * Gets the data types of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @return an array containing objects of the type of each column. The 
     * objects should be Strings, or Doubles.
     */
    public Object [] getKeyTypes() {

	Object [] keyTypes = m_SplitEvaluator.getKeyTypes();
	int numExtraKeys;
	if(m_IsFraction)
	    numExtraKeys = 5;
	else numExtraKeys = 4;
	// Add in the types of our extra fields
	Object [] newKeyTypes = new String [keyTypes.length + numExtraKeys];
	newKeyTypes[0] = new String();
	newKeyTypes[1] = new String();
	newKeyTypes[2] = new String();
	newKeyTypes[3] = new String();
	if(m_IsFraction) newKeyTypes[4] = new String();
	System.arraycopy(keyTypes, 0, newKeyTypes, numExtraKeys, keyTypes.length);
	return newKeyTypes;
    }

    /**
     * Gets the names of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @return an array containing the name of each column
     */
    public String [] getResultNames() {

	String [] resultNames = m_SplitEvaluator.getResultNames();
	// Add in the names of our extra Result fields
	String [] newResultNames = new String [resultNames.length + 3];
	newResultNames[0] = TIMESTAMP_FIELD_NAME;
	newResultNames[1] = SELECTION_TIME_FIELD_NAME;
	newResultNames[2] = COST_FIELD_NAME;
	System.arraycopy(resultNames, 0, newResultNames, 3, resultNames.length);
	return newResultNames;
    }

    /**
     * Gets the data types of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @return an array containing objects of the type of each column. The 
     * objects should be Strings, or Doubles.
     */
    public Object [] getResultTypes() {

	Object [] resultTypes = m_SplitEvaluator.getResultTypes();
	// Add in the types of our extra Result fields
	Object [] newResultTypes = new Object [resultTypes.length + 3];
	newResultTypes[0] = new Double(0);
	newResultTypes[1] = new Double(0);
	newResultTypes[2] = new Double(0);
	System.arraycopy(resultTypes, 0, newResultTypes, 3, resultTypes.length);
	return newResultTypes;
    }

    /**
     * Gets a description of the internal settings of the result
     * producer, sufficient for distinguishing a ResultProducer
    * instance from another with different settings (ignoring
     * those settings set through this interface). For example,
     * a cross-validation ResultProducer may have a setting for the
     * number of folds. For a given state, the results produced should
     * be compatible. Typically if a ResultProducer is an OptionHandler,
     * this string will represent the command line arguments required
     * to set the ResultProducer to that state.
     *
     * @return the description of the ResultProducer state, or null
     * if no state is defined
     */
    public String getCompatibilityState() {

	String result = "-X " + m_NumFolds + " -S " + getStepSize() + 
	    " -L " + getLowerSize() + " -U " + getUpperSize() + " ";
	if (m_SplitEvaluator == null) {
	    result += "<null SplitEvaluator>";
	} else {
	    result += "-W " + m_SplitEvaluator.getClass().getName();
	}
	return result + " --";
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String outputFileTipText() {
	return "Set the destination for saving raw output. If the rawOutput "
	    +"option is selected, then output from the splitEvaluator for "
	    +"individual folds is saved. If the destination is a directory, "
	    +"then each output is saved to an individual gzip file; if the "
	    +"destination is a file, then each output is saved as an entry "
	    +"in a zip file.";
    }

    /**
     * Get the value of OutputFile.
     *
     * @return Value of OutputFile.
     */
    public File getOutputFile() {
    
	return m_OutputFile;
    }
  
    /**
     * Set the value of OutputFile.
     *
     * @param newOutputFile Value to assign to OutputFile.
     */
    public void setOutputFile(File newOutputFile) {
    
	m_OutputFile = newOutputFile;
    }  

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String numFoldsTipText() {
	return "Number of folds to use in cross validation.";
    }

    /**
     * Get the value of NumFolds.
     *
     * @return Value of NumFolds.
     */
    public int getNumFolds() {
    
	return m_NumFolds;
    }
  
    /**
     * Set the value of NumFolds.
     *
     * @param newNumFolds Value to assign to NumFolds.
     */
    public void setNumFolds(int newNumFolds) {
    
	m_NumFolds = newNumFolds;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String lowerSizeTipText() {
	return "Set the minimum number of instances in a training set. Setting zero "
            + "here will actually use <stepSize> number of instances at the first "
            + "step (since performance at zero instances is predictable)";
    }

    /**
     * Get the value of LowerSize.
     *
     * @return Value of LowerSize.
     */
    public double getLowerSize() {
    
	return m_LowerSize;
    }
  
    /**
     * Set the value of LowerSize.
     *
     * @param newLowerSize Value to assign to
     * LowerSize.
     */
    public void setLowerSize(double newLowerSize) {
    
	m_LowerSize = newLowerSize;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String upperSizeTipText() {
	return "Set the maximum number of instances in a training set. Setting -1 "
	    + "sets no upper limit (other than the total number of instances "
	    + "in the full training set)";
    }

    /**
     * Get the value of UpperSize.
     *
     * @return Value of UpperSize.
     */
    public double getUpperSize() {
    
	return m_UpperSize;
    }
  
    /**
     * Set the value of UpperSize.
     *
     * @param newUpperSize Value to assign to
     * UpperSize.
     */
    public void setUpperSize(double newUpperSize) {
    
	m_UpperSize = newUpperSize;
    }


    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String stepSizeTipText() {
	return "Set the number of instances to add to the training data at each step.";
    }

    /**
     * Get the value of StepSize.
     *
     * @return Value of StepSize.
     */
    public double getStepSize() {
    
	return m_StepSize;
    }
  
    /**
     * Set the value of StepSize.
     *
     * @param newStepSize Value to assign to
     * StepSize.
     */
    public void setStepSize(double newStepSize) {
    	m_StepSize = newStepSize;
	//setIsFraction();
    }


    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String plotPointsTipText() {
	return "A list of specific points to plot as a string of numbers separated by commas or spaces. "+
	    "Whole numbers indicate a specific number of examples, "+
	    "decimal fractions indicate a fraction of the total training set. "+
	    "Specifying plot points overrides step size, lower size, and upper size parameters.";
    }

    /**
     * Get the value of PlotPoints.
     *
     * @return Value of PlotPoints.
     */
    public String getPlotPoints() {
	StringBuffer buf = new StringBuffer();
	if (m_PlotPoints != null) 
	    for (int i=0; i < m_PlotPoints.length; i++) {
		buf.append(m_PlotPoints[i]);
		if (i != (m_PlotPoints.length -1)) 
		    buf.append(" ");
	    }
	return buf.toString();
    }
  
    /**
     * Set the value of PlotPoints.
     *
     * @param plotPoints Value to assign to
     * PlotPoints.
     */
    public void setPlotPoints(String plotPoints) {
	m_PlotPoints = parsePlotPoints(plotPoints);
	setIsFraction();
    }

    /** 
     * Parse a string of doubles separated by commas or spaces into a sorted array of doubles
     */
    protected double[] parsePlotPoints(String plotPoints) {
	StringTokenizer tokenizer = new StringTokenizer(plotPoints," ,\t");
	double[] result = null;
	int count = tokenizer.countTokens();
	if (count > 0)
	    result = new double[count];
	else
	    return null;
	int i = 0;
	while(tokenizer.hasMoreTokens()) {
	    result[i] = Double.parseDouble(tokenizer.nextToken());
	    i++;
	}
	Arrays.sort(result);
	return result;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String rawOutputTipText() {
	return "Save raw output (useful for debugging). If set, then output is "
	    +"sent to the destination specified by outputFile";
    }

    /**
     * Get if raw split evaluator output is to be saved
     * @return true if raw split evalutor output is to be saved
     */
    public boolean getRawOutput() {
	return m_debugOutput;
    }
  
    /**
     * Set to true if raw split evaluator output is to be saved
     * @param d true if output is to be saved
     */
    public void setRawOutput(boolean d) {
	m_debugOutput = d;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String splitEvaluatorTipText() {
	return "The evaluator to apply to the cross validation folds. "
	    +"This may be a classifier, regression scheme etc.";
    }
 
    /**
     * Get the SplitEvaluator.
     *
     * @return the SplitEvaluator.
     */
    public SplitEvaluator getSplitEvaluator() {
    
	return m_SplitEvaluator;
    }
  
    /**
     * Set the SplitEvaluator.
     *
     * @param newSplitEvaluator new SplitEvaluator to use.
     */
    public void setSplitEvaluator(SplitEvaluator newSplitEvaluator) {
    
	m_SplitEvaluator = newSplitEvaluator;
	m_SplitEvaluator.setAdditionalMeasures(m_AdditionalMeasures);
    }

    /**
     * Returns an enumeration describing the available options..
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

	Vector newVector = new Vector(8);

	newVector.addElement(new Option(
					"\tThe number of folds to use for the cross-validation.\n"
					+"\t(default 10)", 
					"X", 1, 
					"-X <number of folds>"));

	newVector.addElement(new Option(
					"\tThe number of instances to add at each step on the learning curve.",
					"S", 1, 
					"-S <step size>"));

	newVector.addElement(new Option(
					"\tThe minmum number of instances in a training set. Setting zero"
					+ "\there will actually use <stepSize> number of instances at the first"
					+ "\tstep (since performance at zero instances is predictable)",
					"L", 1, 
					"-L <lower bound>"));

	newVector.addElement(new Option(
					"\tThe maximum number of instances in a training set. Setting -1 "
					+ "\tsets no upper limit (other than the total number of instances "
					+ "\tin the full training set)",
					"U", 1, 
					"-U <upper bound>"));

	newVector.addElement(new Option(
					"\tA list of specific points to plot as a string of numbers\n"+
					"separated by commas or spaces.\n"+
					"Whole numbers indicate a specific number of examples,\n"+
					"decimal fractions indicate a fraction of the total training set.\n"+
					"Specifying plot points overrides the S, L, and U parameters",
 					"P", 1, 
	 				"-P <point list>"));

	newVector.addElement(new Option(
					"Save raw split evaluator output.",
					"D",0,"-D"));

	newVector.addElement(new Option(
					"\tThe filename where raw output will be stored.\n"
					+"\tIf a directory name is specified then then individual\n"
					+"\toutputs will be gzipped, otherwise all output will be\n"
					+"\tzipped to the named file. Use in conjuction with -D."
					+"\t(default splitEvalutorOut.zip)", 
					"O", 1, 
					"-O <file/directory name/path>"));

	newVector.addElement(new Option(
					"\tThe full class name of a SplitEvaluator.\n"
					+"\teg: weka.experiment.ClassifierSplitEvaluator", 
					"W", 1, 
					"-W <class name>"));

	if ((m_SplitEvaluator != null) &&
	    (m_SplitEvaluator instanceof OptionHandler)) {
	    newVector.addElement(new Option(
					    "",
					    "", 0, "\nOptions specific to split evaluator "
					    + m_SplitEvaluator.getClass().getName() + ":"));
	    Enumeration enum = ((OptionHandler)m_SplitEvaluator).listOptions();
	    while (enum.hasMoreElements()) {
		newVector.addElement(enum.nextElement());
	    }
	}
	return newVector.elements();
    }

    /**
     * Parses a given list of options. Valid options are:<p>
     *
     * -X num_folds <br>
     * The number of folds to use for the cross-validation. <p>
     *
     * -D <br>
     * Specify that raw split evaluator output is to be saved. <p>
     *
     * -O file/directory name <br>
     * Specify the file or directory to which raw split evaluator output
     * is to be saved. If a directory is specified, then each output string
     * is saved as an individual gzip file. If a file is specified, then
     * each output string is saved as an entry in a zip file. <p>
     *
     * -W classname <br>
     * Specify the full class name of the split evaluator. <p>
     *
     * All option after -- will be passed to the split evaluator.
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
	setRawOutput(Utils.getFlag('D', options));
	
	String featureDir = Utils.getOption('F', options);
	if (featureDir.length() != 0) {
	    setFeatureCostDirectory(new File(featureDir));
	}
    
	String fName = Utils.getOption('O', options);
	if (fName.length() != 0) {
	    setOutputFile(new File(fName));
	}

	String numFolds = Utils.getOption('X', options);
	if (numFolds.length() != 0) {
	    setNumFolds(Integer.parseInt(numFolds));
	} else {
	    setNumFolds(10);
	}

	String stepSize = Utils.getOption('S', options);
	if (stepSize.length() != 0) {
	    setStepSize(Double.parseDouble(stepSize));
	} else {
	    setStepSize(10);
	}

	String lowerSize = Utils.getOption('L', options);
	if (lowerSize.length() != 0) {
	    setLowerSize(Double.parseDouble(lowerSize));
	} else {
	    setLowerSize(0);
	}
    
	String upperSize = Utils.getOption('U', options);
	if (upperSize.length() != 0) {
	    setUpperSize(Double.parseDouble(upperSize));
	} else {
	    setUpperSize(-1);
	}

	String plotPoints = Utils.getOption('P', options);
	if (plotPoints.length() != 0) {
	    setPlotPoints(plotPoints);
	} else {
	    setPlotPoints("");
	}
	
	String seName = Utils.getOption('W', options);
	if (seName.length() == 0) {
	    throw new Exception("A SplitEvaluator must be specified with"
				+ " the -W option.");
	}
	// Do it first without options, so if an exception is thrown during
	// the option setting, listOptions will contain options for the actual
	// SE.
	setSplitEvaluator((SplitEvaluator)Utils.forName(
							SplitEvaluator.class,
							seName,
							null));
	if (getSplitEvaluator() instanceof OptionHandler) {
	    ((OptionHandler) getSplitEvaluator())
		.setOptions(Utils.partitionOptions(options));
	}
    }

    /**
     * Gets the current settings of the result producer.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {

	String [] seOptions = new String [0];
	if ((m_SplitEvaluator != null) && 
	    (m_SplitEvaluator instanceof OptionHandler)) {
	    seOptions = ((OptionHandler)m_SplitEvaluator).getOptions();
	}

	String [] options = new String [seOptions.length + 17];
	int current = 0;
	
	options[current++] = "-F"; options[current++] = "" + getFeatureCostDirectory();
	options[current++] = "-X"; options[current++] = "" + getNumFolds();

	if (getRawOutput()) {
	    options[current++] = "-D";
	}
	
	options[current++] = "-O"; 
	options[current++] = getOutputFile().getName();
    
	options[current++] = "-S";
	options[current++] = "" + getStepSize();
	options[current++] = "-L";
	options[current++] = "" + getLowerSize();
	options[current++] = "-U";
	options[current++] = "" + getUpperSize();
	options[current++] = "-P";
	options[current++] = getPlotPoints();

	if (getSplitEvaluator() != null) {
	    options[current++] = "-W";
	    options[current++] = getSplitEvaluator().getClass().getName();
	}
	options[current++] = "--";

	System.arraycopy(seOptions, 0, options, current, 
			 seOptions.length);
	current += seOptions.length;
	while (current < options.length) {
	    options[current++] = "";
	}
	return options;
    }

    /**
     * Gets a text descrption of the result producer.
     *
     * @return a text description of the result producer.
     */
    public String toString() {

	String result = "BudgetedLearningCurveResultProducer: ";
	result += getCompatibilityState();
	if (m_Instances == null) {
	    result += ": <null Instances>";
	} else {
	    result += ": " +  Utils.backQuoteChars(m_Instances.relationName());
	}
	return result;
    }

    
    // Quick test 
    public static void main(String [] args) {
	BudgetedLearningCurveResultProducer rp = new BudgetedLearningCurveResultProducer();
	rp.setPlotPoints(args[0]);
	System.out.println(rp.getPlotPoints());
	if (rp.m_PlotPoints != null) System.out.println(isInteger(rp.m_PlotPoints[0]));
    }
} // BudgetedLearningCurveResultProducer




