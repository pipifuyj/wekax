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
 *    NoiseCurveCrossValidationResultProducer.java
 *    
 *    Project
 *    CS 391L Machine Learning
 *
 *    Nishit Shah (nishit@cs.utexas.edu)
 *    
 *
 */

/**
 * Notes:
 * Also Read attached README file
 * Noise will be input as PERCENT ONLY (eg: 10 20 30), No fractions supported
 * The grapher needs integer values of X axis in the learning curve
 * We Take Full Dataset for the runs
 * Use the 4th Key of Fraction as the Noise_Key -- Used Key_Noise_level
 * When we add Noise to a Feature, it does not include the Class attribute
 */

package weka.experiment;
import java.util.*;
import java.io.*;
import weka.experiment.*;
import weka.core.*;
import javax.swing.JComboBox;
import javax.swing.ComboBoxModel;
import javax.swing.DefaultComboBoxModel;


/**
 * Does a N-fold cross-validation, but generates a Noise Curve
 * by also varying the number amount of Noise. Always uses the 
 * same N-fold test set for testing.
 * 
 *
 * @@author Raymond J. Mooney (mooney@@cs.utexas.edu)
 * Changed to plot Noise Curves
 */

public class NoiseCurveCrossValidationResultProducer 
    implements ResultProducer, OptionHandler, AdditionalMeasureProducer {
  
    /** The dataset of interest */
    protected Instances m_Instances;

    /** The ResultListener to send results to */
    protected ResultListener m_ResultListener = new CSVResultListener();

    /** The number of folds in the cross-validation */
    protected int m_NumFolds = 10;

    /** Save raw output of split evaluators --- for debugging purposes */
    protected boolean m_debugOutput = false;
    
    /** Add noise to Class Labels in Training Set */
    protected boolean m_classNoise = true;
    
    /** Add noise to Features, do not include Class as a Feature in Training Set */
    protected boolean m_featureNoise = true;
    
    /** Set features missing, do not include Class as a Feature in Training Set */
    protected boolean m_featureMiss = true;

    /** Add noise to Class Labels in Testing Set */    
    protected boolean m_classNoiseTest = true;

    /** Add noise to Features, do not include Class as a Feature in Testing Set */    
    protected boolean m_featureNoiseTest = true;

    /** Set features missing, do not include Class as a Feature in Testing Set */    
    protected boolean m_featureMissTest = true;
    
    /** The output zipper to use for saving raw splitEvaluator output  */
    protected OutputZipper m_ZipDest = null;

    /** The destination output file/directory for raw output */
    protected File m_OutputFile = new File(
					   new File(System.getProperty("user.dir")), 
					   "splitEvalutorOut.zip");

    /** The SplitEvaluator used to generate results */
    protected SplitEvaluator m_SplitEvaluator = new ClassifierSplitEvaluator();

    /** The names of any additional measures to look for in SplitEvaluators */
    protected String [] m_AdditionalMeasures = null;

    /** Store Statistics of Attributes */
    protected Vector m_AttributeStats = null;

    /** The specific points to plot, either integers representing specific numbers of training examples,
     * or decimal fractions representing percentages of the full training set -- ONLY INTEGERS SUPPORTED*/
    protected double[] m_PlotPoints;

    /** Dataset size for the runs, we take the full dataset*/
    protected int m_CurrentSize = 0;

    /** Random Number, used for randomization in each run*/
    protected Random m_Random = new Random(0);

    /* The name of the key field containing the dataset name */
    public static String DATASET_FIELD_NAME = "Dataset";

    /* The name of the key field containing the run number */
    public static String RUN_FIELD_NAME = "Run";

    /* The name of the key field containing the fold number */
    public static String FOLD_FIELD_NAME = "Fold";

    /* The name of the result field containing the timestamp */
    public static String TIMESTAMP_FIELD_NAME = "Date_time";

    /* The name of the key field containing the learning rate step number */
    public static String STEP_FIELD_NAME = "Total_instances";

    /* The name of the key field containing the fraction of total instances used */
    public static String NOISE_FIELD_NAME = "Noise_levels";

    /**
     * Returns a string describing this result producer
     * @@return a description of the result producer suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
	return "Performs a noise-curve cross validation run using a supplied "
	    +"split evaluator. Trains on different amounts of noise in the Dataset, "
	    +"repeatedly testing on the test set for that split after training.";
    }
    
    /**
     * Sets the dataset that results will be obtained for.
     *
     * @@param instances a value of type 'Instances'.
     */
    public void setInstances(Instances instances) {
    
	m_Instances = instances;
    }

    /**
     * Sets the object to send results of each run to.
     *
     * @@param listener a value of type 'ResultListener'
     */
    public void setResultListener(ResultListener listener) {

	m_ResultListener = listener;
    }

    /**
     * Set a list of method names for additional measures to look for
     * in SplitEvaluators. This could contain many measures (of which only a
     * subset may be produceable by the current SplitEvaluator) if an experiment
     * is the type that iterates over a set of properties.
     * @@param additionalMeasures an array of measure names, null if none
     */
    public void setAdditionalMeasures(String [] additionalMeasures) {
	m_AdditionalMeasures = additionalMeasures;

	if (m_SplitEvaluator != null) {
	    System.err.println("NoiseCurveCrossValidationResultProducer: setting additional "
			       +"measures for "
			       +"split evaluator");
	    m_SplitEvaluator.setAdditionalMeasures(m_AdditionalMeasures);
	}
    }

    /**
     * Returns an enumeration of any additional measure names that might be
     * in the SplitEvaluator
     * @@return an enumeration of the measure names
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
     * @@param measureName the name of the measure to query for its value
     * @@return the value of the named measure
     * @@exception IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {
	if (m_SplitEvaluator instanceof AdditionalMeasureProducer) {
	    return ((AdditionalMeasureProducer)m_SplitEvaluator).
		getMeasure(additionalMeasureName);
	} else {
	    throw new IllegalArgumentException("NoiseCurveCrossValidationResultProducer: "
					       +"Can't return value for : "+additionalMeasureName
					       +". "+m_SplitEvaluator.getClass().getName()+" "
					       +"is not an AdditionalMeasureProducer");
	}
    }
  
    /**
     * Gets a Double representing the current date and time.
     * eg: 1:46pm on 20/5/1999 -> 19990520.1346
     *
     * @@return a value of type Double
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
     * @@exception Exception if an error occurs during preprocessing.
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
     * @@exception Exception if an error occurs
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
     * @@param run the run number to get keys for.
     * @@exception Exception if a problem occurs while getting the keys
     */
    public void doRunKeys(int run) throws Exception {
	int numExtraKeys;
	numExtraKeys = 5;
	if (m_Instances == null) {
	    throw new Exception("No Instances set");
	}
	if (m_ResultListener == null) {
	    throw new Exception("No ResultListener set");
	}
	if (m_PlotPoints  == null) {
	    throw new Exception("Enter atleast one point on Noise Curve");
	}
	for(int noiseLevel = 0; noiseLevel < m_PlotPoints.length; noiseLevel++)
	    {
		for (int fold = 0; fold < m_NumFolds; fold++) {
		    m_CurrentSize = maxTrainSize();
		    // Add in some fields to the key like run and fold number, dataset name
		    Object [] seKey = m_SplitEvaluator.getKey();
		    Object [] key = new Object [seKey.length + numExtraKeys];
		    key[0] = Utils.backQuoteChars(m_Instances.relationName());
		    key[1] = "" + run;
		    key[2] = "" + (fold + 1);
		    key[3] = "" + m_CurrentSize;
		    key[4] = "" + (int) m_PlotPoints[noiseLevel]; //Converting to Integer for Grapher
		    System.arraycopy(seKey, 0, key, numExtraKeys, seKey.length);
		    if (m_ResultListener.isResultRequired(this, key)) {
			try {
			    m_ResultListener.acceptResult(this, key, null);
			} catch (Exception ex) {
			    // Save the train and test datasets for debugging purposes?
			    throw ex;
			}
		    }
		}//for each fold
	    }//for each noise level
    }
    
    /** 
     * Get the maximum size of the training set based on 
     * maximum training set size from the n-fold CV 
     */
    protected int maxTrainSize() {
	return (int)(m_Instances.numInstances()*(1 - 1/((double)m_NumFolds)));
    }

    /**
     * Gets the results for a specified run number. Different run
     * numbers correspond to different randomizations of the data. Results
     * produced should be sent to the current ResultListener
     *
     * @@param run the run number to get results for.
     * @@exception Exception if a problem occurs while getting the results
     */
    public void doRun(int run) throws Exception {
	int numExtraKeys;
	numExtraKeys = 5;
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
	// Check if PlotPoint is Null
	if (m_PlotPoints == null) {
	    throw new Exception("Enter atleast one point on Noise Curve");
	}
	m_AttributeStats = new Vector(m_Instances.numAttributes());	
	//Storing both Nominal and  Numeric attributes, we use Numeric values for finding Mean and Variance
	for (int i = 0; i<m_Instances.numAttributes(); i++)
	    {
		if(m_Instances.attribute(i).isNominal()){
		int []nomCounts = (m_Instances.attributeStats(i)).nominalCounts;
		double []counts = new double[nomCounts.length];
		double []stats = new double[counts.length - 1];
		stats[0] = counts[0];
		//Calculate cumulative probabilities
		for(int j=1; j<stats.length; j++)
		    stats[j] = stats[j-1] + counts[j];
		m_AttributeStats.add(i,stats);
		}
		if(m_Instances.attribute(i).isNumeric())
		  {
			double []stats = new double[2];
			stats[0] = m_Instances.meanOrMode(i);
			stats[1] = Math.sqrt(m_Instances.variance(i));
			m_AttributeStats.add(i, stats);
			  }
	    }
	//Initialize Random Number, The experiment will be repeatable for the same run number
	m_Random = new Random(run);
	// Randomize on a copy of the original dataset
	Instances runInstances = new Instances(m_Instances);
	runInstances.randomize(new Random(run));
	if (runInstances.classAttribute().isNominal()) {
	    runInstances.stratify(m_NumFolds);
	}
	//For Each Noise Level
	for (int noiseLevel = 0; noiseLevel < m_PlotPoints.length; noiseLevel++)
	    {
		System.out.println("\n\nRun : " + run + " Number of Noise Levels : " + m_PlotPoints.length + " Noise Level : " + m_PlotPoints[noiseLevel] + "\n");
		for (int fold = 0; fold < m_NumFolds; fold++) {
		    Instances train = runInstances.trainCV(m_NumFolds, fold);
		    // Randomly shuffle stratified training set for fold: added by Sugato
		    train.randomize(new Random(fold));	    
		    Instances test = runInstances.testCV(m_NumFolds, fold);
		    if (m_classNoise == true){
			addClassNoise(train, test, noiseLevel);
			//System.out.println("Hi");
		    }
		    // Check m_featureNoise, if true call addFeatureNoise(train, test)
		    if (m_featureNoise == true){
			addFeatureNoise(train, test, noiseLevel);
			//System.out.println("Hi");
		    }
		    // Check m_featureMiss, if true call addFeatureMiss(train, test)
		    if (m_featureMiss == true){
			addFeatureMiss(train, test, noiseLevel);
			//System.out.println("Hi");
		    }
		    m_CurrentSize = maxTrainSize();
		    // Add in some fields to the key like run and fold number, dataset name
		    Object [] seKey = m_SplitEvaluator.getKey();
		    Object [] key = new Object [seKey.length + numExtraKeys];
		    key[0] = Utils.backQuoteChars(m_Instances.relationName());
		    key[1] = "" + run;
		    key[2] = "" + (fold + 1);
		    key[3] = "" + m_CurrentSize;
		    key[4] = "" + (int)m_PlotPoints[noiseLevel];
		    System.arraycopy(seKey, 0, key, numExtraKeys, seKey.length);
		    if (m_ResultListener.isResultRequired(this, key)) {
			try {
			    System.out.println("Run:" + run + " Fold:" + fold + " Size:" + m_CurrentSize + " Noise Level:" + m_PlotPoints[noiseLevel]);
			    Instances trainSubset = new Instances(train, 0, m_CurrentSize);
			    Object [] seResults = m_SplitEvaluator.getResult(trainSubset, test);
			    Object [] results = new Object [seResults.length + 1];
			    results[0] = getTimestamp();
			    System.arraycopy(seResults, 0, results, 1,
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
		}//Number of Folds
	    }//For each Noise Level
    }

    /** Return the amount of noise for the ith point on the
     * curve for plotPoints as specified. Percent of NOISE Returned
     * Can Simplify this procedure to return m_PlotPoints[i] directly
     */
    protected int plotPoint(int i) {
	// If i beyond number of given plot points return a value greater than maximum training size
	if (i >= m_PlotPoints.length)
	    return maxTrainSize() + 1;
	double point = m_PlotPoints[i];
	// If plot point is an integer (other than a non-initial 1)
	// treat it as a specific number of examples
	if (isInteger(point) && !(Utils.eq(point, 1.0) && i!=0))
	    return (int)point;
	else
	    // Otherwise, treat it as a percentage of the full set
	    return (int)Math.round(point * maxTrainSize());
    }

    /** Return true if the given double represents an integer value */
    protected static boolean isInteger(double val) {
	return Utils.eq(Math.floor(val), Math.ceil(val));
    }

    /**
     * Gets the names of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @@return an array containing the name of each column
     */
    public String [] getKeyNames() {

	String [] keyNames = m_SplitEvaluator.getKeyNames();
	// Add in the names of our extra key fields
	int numExtraKeys;
	numExtraKeys = 5;
	String [] newKeyNames = new String [keyNames.length + numExtraKeys];
	newKeyNames[0] = DATASET_FIELD_NAME;
	newKeyNames[1] = RUN_FIELD_NAME;
	newKeyNames[2] = FOLD_FIELD_NAME;
	newKeyNames[3] = STEP_FIELD_NAME;
	newKeyNames[4] = NOISE_FIELD_NAME;
	System.arraycopy(keyNames, 0, newKeyNames, numExtraKeys, keyNames.length);
	return newKeyNames;
    }

    /**
     * Gets the data types of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @@return an array containing objects of the type of each column. The 
     * objects should be Strings, or Doubles.
     */
    public Object [] getKeyTypes() {

	Object [] keyTypes = m_SplitEvaluator.getKeyTypes();
	int numExtraKeys;
	numExtraKeys = 5;
	// Add in the types of our extra fields
	Object [] newKeyTypes = new String [keyTypes.length + numExtraKeys];
	newKeyTypes[0] = new String();
	newKeyTypes[1] = new String();
	newKeyTypes[2] = new String();
	newKeyTypes[3] = new String();
	newKeyTypes[4] = new String();
	System.arraycopy(keyTypes, 0, newKeyTypes, numExtraKeys, keyTypes.length);
	return newKeyTypes;
    }

    /**
     * Gets the names of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @@return an array containing the name of each column
     */
    public String [] getResultNames() {

	String [] resultNames = m_SplitEvaluator.getResultNames();
	// Add in the names of our extra Result fields
	String [] newResultNames = new String [resultNames.length + 1];
	newResultNames[0] = TIMESTAMP_FIELD_NAME;
	System.arraycopy(resultNames, 0, newResultNames, 1, resultNames.length);
	return newResultNames;
    }

    /**
     * Gets the data types of each of the columns produced for a single run.
     * This method should really be static.
     *
     * @@return an array containing objects of the type of each column. The 
     * objects should be Strings, or Doubles.
     */
    public Object [] getResultTypes() {

	Object [] resultTypes = m_SplitEvaluator.getResultTypes();
	// Add in the types of our extra Result fields
	Object [] newResultTypes = new Object [resultTypes.length + 1];
	newResultTypes[0] = new Double(0);
	System.arraycopy(resultTypes, 0, newResultTypes, 1, resultTypes.length);
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
     * @@return the description of the ResultProducer state, or null
     * if no state is defined
     */
    public String getCompatibilityState() {
	//Modify for Noise
	String result = "-X " + m_NumFolds;
	if (m_SplitEvaluator == null) {
	    result += "<null SplitEvaluator>";
	} else {
	    result += "-W " + m_SplitEvaluator.getClass().getName();
	}
	return result + " --";
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
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
     * @@return Value of OutputFile.
     */
    public File getOutputFile() {
    
	return m_OutputFile;
    }
  
    /**
     * Set the value of OutputFile.
     *
     * @@param newOutputFile Value to assign to OutputFile.
     */
    public void setOutputFile(File newOutputFile) {
    
	m_OutputFile = newOutputFile;
    }  

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String numFoldsTipText() {

	return "Number of folds to use in cross validation.";
    }

    /**
     * Get the value of NumFolds.
     *
     * @@return Value of NumFolds.
     */
    public int getNumFolds() {
    
	return m_NumFolds;
    }
  
    /**
     * Set the value of NumFolds.
     *
     * @@param newNumFolds Value to assign to NumFolds.
     */
    public void setNumFolds(int newNumFolds) {
    
	m_NumFolds = newNumFolds;
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String plotPointsTipText() {

	return "A list of specific points (Whole Numbers) to plot as a string of numbers separated by commas or spaces. "+
	    "Whole numbers indicate a specific Percentages of Noise ";
    }

    /**
     * Get the value of PlotPoints.
     *
     * @@return Value of PlotPoints.
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
     * @@param plotPoints Value to assign to
     * PlotPoints.
     */
    public void setPlotPoints(String plotPoints) {
	m_PlotPoints = parsePlotPoints(plotPoints);
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
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String rawOutputTipText() {
	return "Save raw output (useful for debugging). If set, then output is "
	    +"sent to the destination specified by outputFile";
    }

    /**
     * Get if raw split evaluator output is to be saved
     * @@return true if raw split evalutor output is to be saved
     */
    public boolean getRawOutput() {
	return m_debugOutput;
    }
  
    /**
     * Set to true if raw split evaluator output is to be saved
     * @@param d true if output is to be saved
     */
    public void setRawOutput(boolean d) {
	m_debugOutput = d;
    }


    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String classNoiseTipText() {
	return "Add Noise to Class";
    }

    /**
     * Get if Noise is to be added to Class
     * @@return true if Noise is to be added to Class
     */
    public boolean getclassNoise() {
	return m_classNoise;
    }

    /**
     * Set to true if Noise is to be added to Class
     * @@param d true if Noise is to be added to Class
     */
    public void setclassNoise(boolean d) {
	m_classNoise = d;
       	if (d == false)
	    {
		setclassNoiseTest(false);
	    }
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String featureNoiseTipText() {
	return "Add Noise to feature";
    }

    /**
     * Get if Noise to be added in Features
     * @@return true if  Noise to be added in Features
     */
    public boolean getfeatureNoise() {
	return m_featureNoise;
    }

    /**
     * Set to true if Noise is to be added to Features
     * @@param d true if Noise is to be added to Features
     */
    public void setfeatureNoise(boolean d) {
	m_featureNoise = d;
	if (d == false)
	    {
	    setfeatureNoiseTest(false);
	    }
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String featureMissTipText() {
	return "Add Missing feature";
    }

    /**
     * Get if Features are to be set Missing
     * @@return true if  Features are to be set Missing
     */
    public boolean getfeatureMiss() {
	return m_featureMiss;
    }

    /**
     * Set to true if Features are to be set Missing
     * @@param d true if Features are to be set Missing
     */
    public void setfeatureMiss(boolean d) {
	m_featureMiss = d;
	if (d == false)
	    {
		setfeatureMissTest(false);
	    }
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String classNoiseTestTipText() {
	return "Add Noise to Class while Testing";
    }

    /**
     * Get if Noise is to be added to Class in Testing Set
     * @@return true if Noise is to be added to Class in Testing Set
     */
    public boolean getclassNoiseTest() {
	return m_classNoiseTest;
    }

    /**
     * Set to true if Noise is to be added to Class in Testing
     * @@param d true if Noise is to be added to Class in Testing
     */
    public void setclassNoiseTest(boolean d) {
	m_classNoiseTest = d;
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String featureNoiseTestTipText() {
	return "Add Noise to feature while Testing";
    }

    /**
     * Get if Noise is to be added to Feature in Testing Set
     * @@return true if Noise is to be added to Feature in Testing Set
     */
    public boolean getfeatureNoiseTest() {
	return m_featureNoiseTest;
    }

    /**
     * Set to true if Noise is to be added in Fetures in Testing
     * @@param d true if Noise is to be added in Fetures in Testing
     */
    public void setfeatureNoiseTest(boolean d) {
	m_featureNoiseTest = d;
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String featureMissTestTipText() {
	return "Add Missing feature to Testing";
    }

    /**
     * Get if Features are to be set Missing in Testing Set
     * @@return true if Features are to be set Missing in Testing Set
     */
    public boolean getfeatureMissTest() {
	return m_featureMissTest;
    }

    /**
     * Set to true if Features are to be set Missing in Testing
     * @@param d true if Features are to be set Missing in Testing
     */
    public void setfeatureMissTest(boolean d) {
	m_featureMissTest = d;
    }

    /**
     * Returns the tip text for this property
     * @@return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String splitEvaluatorTipText() {
	return "The evaluator to apply to the cross validation folds. "
	    +"This may be a classifier, regression scheme etc.";
    }
 
    /**
     * Get the SplitEvaluator.
     *
     * @@return the SplitEvaluator.
     */
    public SplitEvaluator getSplitEvaluator() {
    
	return m_SplitEvaluator;
    }
  
    /**
     * Set the SplitEvaluator.
     *
     * @@param newSplitEvaluator new SplitEvaluator to use.
     */
    public void setSplitEvaluator(SplitEvaluator newSplitEvaluator) {
    
	m_SplitEvaluator = newSplitEvaluator;
	m_SplitEvaluator.setAdditionalMeasures(m_AdditionalMeasures);
    }

    /**
     * Returns an enumeration describing the available options..
     *
     * @@return an enumeration of all the available options.
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(10);
	newVector.addElement(new Option(
					"\tThe number of folds to use for the cross-validation.\n"
					+"\t(default 10)", 
					"X", 1, 
					"-X <number of folds>"));
	newVector.addElement(new Option(
					"\tA list of specific points to plot as a string of numbers\n"+
					"separated by commas or spaces.\n"+
					"Whole numbers indicate a specific number of examples,\n",
 					"P", 1, 
	 				"-P <point list>"));

	newVector.addElement(new Option(
					"Save raw split evaluator output.",
					"D",0,"-D"));
	newVector.addElement(new Option(
					"Noise add to Class in Training.",
					"N",0,"-N"));
	newVector.addElement(new Option(
					"Noise add to Feature in Training.",
					"F",0,"-F"));
	newVector.addElement(new Option(
					"Set Features Missing in Training.",
					"M",0,"-M"));
	newVector.addElement(new Option(
					"Noise add to Class in Testing (Overridden if Option -N not selected).",
					"n",0,"-n"));
	newVector.addElement(new Option(
					"Noise add to Feature in Testing (Overridden if Option -F not selected).",
					"f",0,"-f"));
	newVector.addElement(new Option(
					"Set Features Missing in Testing (Overridden if Option -M not selected).",
					"m",0,"-m"));

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
     * -N Add Noise to Class in Training
     * -n Add Noise to Class in Testing
     * -F Add Noise to Features in Training
     * -f Add Noise to Features in Testing
     * -M Set Features Missing in Training
     * -m Set Features Missing in Testing
     *
     * All option after -- will be passed to the split evaluator.
     *
     * @@param options the list of options as an array of strings
     * @@exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
    
	setRawOutput(Utils.getFlag('D', options));
	//First we set the Test parameters, if the Train parameters are false, they will in the later set turn these parameters off to override
	setclassNoiseTest(Utils.getFlag('n', options));
	setfeatureNoiseTest(Utils.getFlag('f', options));
	setfeatureMissTest(Utils.getFlag('m', options));
	setclassNoise(Utils.getFlag('N', options));
	setfeatureNoise(Utils.getFlag('F', options));
	setfeatureMiss(Utils.getFlag('M', options));

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
     * @@return an array of strings suitable for passing to setOptions
     */
    public String [] getOptions() {

	String [] seOptions = new String [0];
	if ((m_SplitEvaluator != null) && 
	    (m_SplitEvaluator instanceof OptionHandler)) {
	    seOptions = ((OptionHandler)m_SplitEvaluator).getOptions();
	}
	//CHECK VALUE OF seOptions.length + 16 + 4 - 6 + 8
	String [] options = new String [seOptions.length + 20 + 8];
	int current = 0;

	options[current++] = "-X"; options[current++] = "" + getNumFolds();

	if (getRawOutput()) {
	    options[current++] = "-D";
	}

	if (getclassNoise()) {
	    options[current++] = "-N";
	}
	if (getfeatureNoise()) {
	    options[current++] = "-F";
	}
	if (getfeatureMiss()) {
	    options[current++] = "-M";
	}

	if (getclassNoiseTest()) {
	    options[current++] = "-n";
	}
	if (getfeatureNoiseTest()) {
	    options[current++] = "-f";
	}
	if (getfeatureMissTest()) {
	    options[current++] = "-m";
	}

	options[current++] = "-O"; 
	options[current++] = getOutputFile().getName();
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
     * @@return a text description of the result producer.
     */
    public String toString() {

	String result = "NoiseCurveCrossValidationResultProducer: ";
	result += getCompatibilityState();
	if (m_Instances == null) {
	    result += ": <null Instances>";
	} else {
	    result += ": " +  Utils.backQuoteChars(m_Instances.relationName());
	}
	return result;
    }

    public void addClassNoise(Instances train, Instances test, int noiseLevel) throws Exception {
	Instance curr;
	double noisePercent = m_PlotPoints[noiseLevel];
	//The experiment will be repeatable for a particular run
	//Adding to Training
	//Method 1 - Did not use this
	// For each instance, toss a coin, determine whether to change the class
	// If decided to change, randomly select from the range of class calues
	//Method 2 - Used this
	// Find number (N) of instances to add noise to
	// Loop for N
	// Randomly choose an instance
	// Randomly choose class label
	// Assign new class label
	double N1 = (noisePercent / 100.0);
	double N2 = train.numInstances() * N1;
	int N = (int) N2;
	for (int i = 0; i < N; i++){
	    //Pick an instance randomly
	    int instanceNumber = (int)(m_Random.nextDouble() * train.numInstances());
	    //Instance pointed by current curr
	    curr = train.instance(instanceNumber);
	    //Toss a coin and change if Class Label is Nominal
	    if (train.classAttribute().isNominal()){
		int oldClass = (int)train.instance(instanceNumber).classValue();
		int newClass = (int)(m_Random.nextDouble() * train.numClasses());
		while (newClass == oldClass)
		    {
			newClass = (int)(m_Random.nextDouble() * train.numClasses());
		    }
		curr.setClassValue(newClass);
	    }
	    else{
		System.out.println("CLASS NOT NOMINAL");
	    }
	}//end of adding noise to train class
	    
	//If true m_classNoiseTest Adding to Testing	
	if (m_classNoiseTest){
	    double N3 = (noisePercent / 100.0);
	    double N4 = test.numInstances() * N3;
	    int NN = (int) N4;
	    for (int i = 0; i < NN; i++){
		//Pick an instance randomly
		int instanceNumber = (int)(m_Random.nextDouble() * test.numInstances());
		//Instance pointed by current curr
		curr = test.instance(instanceNumber);
		//Toss a coin and change if Class Label is Nominal
		if (test.classAttribute().isNominal()){
		    curr.setClassValue((int)(m_Random.nextDouble() * test.numClasses()));
		}
		else{
		    System.out.println(" CLASS NOT NOMINAL");
		}
	    }
	}//end of adding noise to test class	
    }//end of classnoise

    public void addFeatureNoise(Instances train, Instances test, int noiseLevel) throws Exception {

	//Add noise to Training Set
	//Find number of Features (N) to be added noise to
	Instance curr;
	double noisePercent = m_PlotPoints[noiseLevel];
	double N1 = (noisePercent / 100.0);
	double N2 = train.numInstances() * (train.numAttributes()-1) * N1;
	int N = (int) N2;
	//Loop for N
	for (int i = 0; i < N; i++)
	    {		
		//Pick a random instance
		int instanceNumber = (int)(m_Random.nextDouble() * train.numInstances());
		curr = train.instance(instanceNumber);
		//Pick this Instance's feature randomly
		int attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
		double attValue = 0.0;
		if (attIndex == curr.classIndex())
		    {
			while (attIndex == curr.classIndex()){
			    attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
			}			
		    }
		//Check if Nominal add noise randomly
		if (curr.attribute(attIndex).isNominal())
		    {
			attValue = (double)(int)(m_Random.nextDouble() * curr.attribute(attIndex).numValues());
			curr.setValue(attIndex, (double)attValue);
		    }
		//If Numeric find mean, standard Deviation, find Gaussian value
		if (curr.attribute(attIndex).isNumeric())
		    {
			double []stats = (double [])m_AttributeStats.get(attIndex);
			attValue = (double)((m_Random.nextGaussian() * stats[1]) + stats[0]);
			attValue = Utils.roundDouble((double)attValue, 2);
			curr.setValue(attIndex, (double)attValue);
			// For Checking Trace::: System.out.println("Feature, Train, Numeric, Total Noisey" + N + "Instance Num: " + instanceNumber + "AttIndex: " + attIndex + "Value : " + attValue + "Hence: " + curr.toString());
		    }
	    }
	//If true m_featureNoiseTest, Add noise to Test Set
	if (m_featureNoiseTest == true)
	    {		
		//Find number of Features (NN) to be added noise to
		double N3 = (noisePercent / 100.0);
		double N4 = test.numInstances() * (test.numAttributes()-1) * N3;
		int NN = (int) N4;
		//Loop for NN
		for (int i = 0; i < NN; i++)
		    {		
			//Pick a random instance
			int instanceNumber = (int)(m_Random.nextDouble() * test.numInstances());
			curr = test.instance(instanceNumber);
			//Pick this Instance's feature randomly
			int attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
			double attValue = 0.0;
			if (attIndex == curr.classIndex())
			    {
				while (attIndex == curr.classIndex()){
				    attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
				}
			    }
			//Check if Nominal add noise randomly
			if (curr.attribute(attIndex).isNominal())
			    {
				attValue = (double)(int)(m_Random.nextDouble() * curr.attribute(attIndex).numValues());
				curr.setValue(attIndex, (double)attValue);
				//For Checking Trace::: System.out.println("Feature, Nominal, Testing, Total Noisey: " + NN + ": Instance Num: " + instanceNumber + "AttIndex: " + attIndex + "Value : " + attValue + ": " + curr.toString());
			    }
			//If Numeric find mean, standard Deviation, find Gaussian value
			if (curr.attribute(attIndex).isNumeric())
			    {
				double []stats = (double [])m_AttributeStats.get(attIndex);
				attValue = (double)((m_Random.nextGaussian() * stats[1]) + stats[0]);
				attValue = Utils.roundDouble((double)attValue, 2);
				curr.setValue(attIndex, (double)attValue);
			    }
		    }
	    }//end of feature noise test
    }//End of Feature Noise
    
    public void addFeatureMiss(Instances train, Instances test, int noiseLevel) throws Exception {
	//IF WE WANT TO INCLUDE MISSING IN CLASS THEN REMOVE -1 from N2 calculation and remove while loop
	//for Training set
	//find number of feature to make missing
	Instance curr;
	double noisePercent = m_PlotPoints[noiseLevel];
	double N1 = (noisePercent / 100.0);
	double N2 = train.numInstances() * (train.numAttributes()-1) * N1;
	int N = (int) N2;
	//Loop for number of feature to make missing
	for (int i = 0; i < N; i++)
	    {		
		//Pick a random instance   
		int instanceNumber = (int)(m_Random.nextDouble() * train.numInstances());
		curr = train.instance(instanceNumber);
		//Pick this Instance's feature, randomly choose an attribute
		int attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
		if (attIndex == curr.classIndex())
		    {
			while (attIndex == curr.classIndex()){
			    attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
			}			
		    }
		//Set it missing
		//For Checking Trace::: System.out.println("Train, Total Missing " + N + " Setting instance" + instanceNumber + "Feature " + attIndex + "Missing");
		curr.setMissing(attIndex);
	    }

      	//If true, m_featureMissTest,  for test set
	if (m_featureMissTest == true)
	    {
		double N3 = (noisePercent / 100.0);
		double N4 = test.numInstances() * (test.numAttributes()-1) * N3;
		int NN = (int) N4;
		//Loop for number of feature to make missing
		for (int i = 0; i < NN; i++)
		    {		
			//Pick a random instance   
			int instanceNumber = (int)(m_Random.nextDouble() * test.numInstances());
			curr = test.instance(instanceNumber);
			//Pick this Instance's feature randomly	//randomly choose an attribute
			int attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
			if (attIndex == curr.classIndex())
			    {
				while (attIndex == curr.classIndex()){
				    attIndex = (int)(m_Random.nextDouble() * curr.numAttributes());
				}			
			    }
			//Set it missing
			//For Checking Trace::: System.out.println("Testing, Total Missing " + NN + "Setting instance" + instanceNumber + "Feature " + attIndex + "Missing");
			curr.setMissing(attIndex);
		    }
	    }
    }//End of Feature Miss

    // Quick test 
    public static void main(String [] args) {
	NoiseCurveCrossValidationResultProducer rp = new NoiseCurveCrossValidationResultProducer();
	rp.setPlotPoints(args[0]);
	System.out.println(rp.getPlotPoints());
	if (rp.m_PlotPoints != null) System.out.println(isInteger(rp.m_PlotPoints[0]));
    }
} // NoiseCurveCrossValidationResultProducer





