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
 *    Crate.java
 *    Copyright (C) 2002 Prem Melville
 *
 */

//!! WARNING: Under Development !!

package weka.classifiers.meta;
import weka.classifiers.*;
import java.util.*;
import weka.core.*;
import weka.experiment.*;

/**
 * CRATE (Committee Regressor using Artificial Training Examples) is a
 * meta-learner for building diverse ensembles of regressors by
 * adding specially constructed artificial training
 * examples. Comprehensive experiments have demonstrated that this
 * technique is consistently more accurate than bagging and more
 * accurate that boosting when training data is limited. For more
 * details see <p>
 *
 * Prem Melville and Raymond J. Mooney. <i>Constructing diverse
 * classifier ensembles using artificial training examples.</i>
 * Proceedings of the Seventeeth International Joint Conference on
 * Artificial Intelligence 2003.<BR><BR>
 *
 * Valid options are:<p>
 *
 * -D <br>
 * Turn on debugging output.<p>
 *
 * -W classname <br>
 * Specify the full class name of a weak classifier as the basis for 
 * Crate (default weka.classifiers.trees.j48.J48()).<p>
 *
 * -I num <br>
 * Specify the desired size of the committee (default 15). <p>
 *
 * -M iterations <br>
 * Set the maximum number of Crate iterations (default 50). <p>
 *
 * -S seed <br>
 * Seed for random number generator. (default 0).<p>
 *
 * -R factor <br>
 * Factor that determines number of artificial examples to generate. <p>
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Prem Melville (melville@cs.utexas.edu)
 */
public class Crate extends Classifier implements OptionHandler{

    /** Set to true to get debugging output. */
    protected boolean m_Debug = true;

    /** The model base classifier to use. */
    protected Classifier m_Classifier = new weka.classifiers.trees.m5.M5P();
      
    /** Vector of classifiers that make up the committee/ensemble. */
    protected Vector m_Committee = null;
    
    /** The desired ensemble size. */
    protected int m_DesiredSize = 25;

    /** The maximum number of Crate iterations to run. */
    protected int m_NumIterations = 150;
    
    /** The seed for random number generation. */
    protected int m_Seed = 0;
    
    /** Amount of artificial/random instances to use - specified as a
        fraction of the training data size. */
    protected double m_ArtSize = 1.0 ;

    /** The random number generator. */
    protected Random m_Random = new Random(0);
    
    /** Attribute statistics - used for generating artificial examples. */
    protected Vector m_AttributeStats = null;
    
    /** Factor specifying desired amount of diversity */
    protected double m_Alpha = 1.5;

    /** Evaluator */
    protected Evaluation m_Evaluation; 
    
    /** Choice of error measure to optimize for */
    static final int RMS = 0,
	MAE = 1,
	ROOT_RELATIVE_SQUARED = 2;
    
    /** Error measure to optimize for */
    protected int m_ErrorMeasure = RMS;
    
    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(10);
	newVector.addElement(new Option(
	      "\tTurn on debugging output.",
	      "D", 0, "-D"));
	newVector.addElement(new Option(
              "\tDesired size of ensemble.\n" 
              + "\t(default 15)",
              "I", 1, "-I"));
	newVector.addElement(new Option(
	      "\tMaximum number of Crate iterations.\n"
	      + "\t(default 50)",
	      "M", 1, "-M"));
	newVector.addElement(new Option(
	      "\tFull name of base classifier.\n"
	      + "\t(default weka.classifiers.trees.j48.J48)",
	      "W", 1, "-W"));
	newVector.addElement(new Option(
              "\tSeed for random number generator.\n"
	      +"\tIf set to -1, use a random seed.\n"
              + "\t(default 0)",
              "S", 1, "-S"));
	newVector.addElement(new Option(
              "\tFactor specifying desired amount of diversity.\n"
	      + "\t(default 1.5)",
              "V", 1, "-V"));
	newVector.addElement(new Option(
				    "\tFactor that determines number of artificial examples to generate.\n"
				    +"\tSpecified proportional to training set size.\n" 
				    + "\t(default 1.0)",
				    "R", 1, "-R"));
    	newVector.addElement(new Option(
	       "\tError measure to evaluate for.\n"
	      +"\t0=RMS, 1=MAE, 2=Root Relative Squared Error\n" 
	      + "\t(default 0)",
	      "E", 1, "-E"));
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
     * -D <br>
     * Turn on debugging output.<p>
     *    
     * -W classname <br>
     * Specify the full class name of a weak classifier as the basis for 
     * Crate (required).<p>
     *
     * -I num <br>
     * Specify the desired size of the committee (default 15). <p>
     *
     * -M iterations <br>
     * Set the maximum number of Crate iterations (default 50). <p>
     *
     * -S seed <br>
     * Seed for random number generator. (default 0).<p>
     *
     * -R factor <br>
     * Factor that determines number of artificial examples to generate. <p>
     *
     * Options after -- are passed to the designated classifier.<p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
	setDebug(Utils.getFlag('D', options));

	String desiredSize = Utils.getOption('I', options);
	if (desiredSize.length() != 0) {
	    setDesiredSize(Integer.parseInt(desiredSize));
	} else {
	    setDesiredSize(15);
	}

	String maxIterations = Utils.getOption('M', options);
	if (maxIterations.length() != 0) {
	    setNumIterations(Integer.parseInt(maxIterations));
	} else {
	    setNumIterations(50);
	}
	
	String seed = Utils.getOption('S', options);
	if (seed.length() != 0) {
	    setSeed(Integer.parseInt(seed));
	} else {
	    setSeed(0);
	}
	
	String artSize = Utils.getOption('R', options);
	if (artSize.length() != 0) {
	    setArtificialSize(Double.parseDouble(artSize));
	} else {
	    setArtificialSize(1.0);
	}
	
	String alpha = Utils.getOption('V', options);
	if (alpha.length() != 0) {
	    setAlpha(Double.parseDouble(alpha));
	} else {
	    setAlpha(1.5);
	}
	
	String errorMeasure = Utils.getOption('E', options);
	if (errorMeasure.length() != 0) {
	    setErrorMeasure(Integer.parseInt(errorMeasure));
	} else {
	    setErrorMeasure(0);
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
	String [] options = new String [classifierOptions.length + 16];
	int current = 0;
	if (getDebug()) {
	    options[current++] = "-D";
	}
	options[current++] = "-S"; options[current++] = "" + getSeed();
	options[current++] = "-I"; options[current++] = "" + getDesiredSize();
	options[current++] = "-M"; options[current++] = "" + getNumIterations();
	options[current++] = "-R"; options[current++] = "" + getArtificialSize();
	options[current++] = "-V"; options[current++] = "" + getAlpha();
	options[current++] = "-E"; options[current++] = "" + getErrorMeasure();

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
     * Get the value of errorMeasure.
     * @return value of errorMeasure.
     */
    public int getErrorMeasure() {
	return m_ErrorMeasure;
    }
    
    /**
     * Set the value of errorMeasure.
     * @param v  Value to assign to errorMeasure.
     */
    public void setErrorMeasure(int  v) {
	m_ErrorMeasure = v;
    }
    
    /**
     * Get the value of Alpha.
     * @return value of Alpha.
     */
    public double getAlpha() {
	return m_Alpha;
    }
    
    /**
     * Set the value of Alpha.
     * @param v  Value to assign to Alpha.
     */
    public void setAlpha(double  v) {
	m_Alpha = v;
    }
    
    
    /**
     * Set debugging mode
     *
     * @param debug true if debug output should be printed
     */
    public void setDebug(boolean debug) {
	m_Debug = debug;
    }
    
    /**
     * Get whether debugging is turned on
     *
     * @return true if debugging output is on
     */
    public boolean getDebug() {
	return m_Debug;
    }
    
    /**
     * Set the base classifier for Crate.
     *
     * @param newClassifier the Classifier to use.
     */
    public void setClassifier(Classifier newClassifier) {
	m_Classifier = newClassifier;
    }

    /**
     * Get the classifier used as the base classifier
     *
     * @return the classifier used as the classifier
     */
    public Classifier getClassifier() {
	return m_Classifier;
    }

    /**
     * Factor that determines number of artificial examples to generate.
     *
     * @return factor that determines number of artificial examples to generate
     */
    public double getArtificialSize() {
	return m_ArtSize;
    }
  
    /**
     * Sets factor that determines number of artificial examples to generate.
     *
     * @param newwArtSize factor that determines number of artificial examples to generate
     */
    public void setArtificialSize(double newArtSize) {
	m_ArtSize = newArtSize;
    }
    
    /**
     * Gets the desired size of the committee.
     *
     * @return the desired size of the committee
     */
    public int getDesiredSize() {
	return m_DesiredSize;
    }
    
    /**
     * Sets the desired size of the committee.
     *
     * @param newDesiredSize the desired size of the committee
     */
    public void setDesiredSize(int newDesiredSize) {
	m_DesiredSize = newDesiredSize;
    }
    
    /**
     * Sets the max number of Crate iterations to run.
     *
     * @param numIterations  max number of Crate iterations to run
     */
    public void setNumIterations(int numIterations) {
	m_NumIterations = numIterations;
    }

    /**
     * Gets the max number of Crate iterations to run.
     *
     * @return the  max number of Crate iterations to run
     */
    public int getNumIterations() {
        return m_NumIterations;
    }
    
    /**
     * Set the seed for random number generator.
     *
     * @param seed the random number seed 
     */
    public void setSeed(int seed) {
	m_Seed = seed;
	if(m_Seed==-1){
	    m_Random = new Random();
	}else{
	    m_Random = new Random(m_Seed);
	}
    }
    
    /**
     * Gets the seed for the random number generator.
     *
     * @return the seed for the random number generator
     */
    public int getSeed() {
        return m_Seed;
    }

    /**
     * Build Crate classifier
     *
     * @param data the training data to be used for generating the classifier
     * @exception Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
	if(m_Classifier == null) {
	    throw new Exception("A base classifier has not been specified!");
	}
	if(data.checkForStringAttributes()) {
	    throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
	}
	if(!(data.classAttribute().isNumeric())) {
	    throw new UnsupportedClassTypeException("Crate must be applied to numeric classes!");
	}
	if(m_NumIterations < m_DesiredSize)
	    throw new Exception("Max number of iterations must be >= desired ensemble size!");
	
	int i = 1;//current committee size
	int numTrials = 1;//number of Crate iterations 
	Instances divData = new Instances(data);//local copy of data - diversity data
	divData.deleteWithMissingClass();
	//m_Evaluation = new Evaluation(divData);
	Instances artData = null;//artificial data

	//compute number of artficial instances to add at each iteration
	int artSize = (int) (Math.abs(m_ArtSize)*divData.numInstances());
	if(artSize==0) artSize=1;//atleast add one random example
	computeStats(data);//Compute training data stats for creating artificial examples
	
	//initialize new committee
	m_Committee = new Vector();
	Classifier copiesOfClassifier[] = Classifier.makeCopies(m_Classifier,m_DesiredSize);
	//All these copies may not be used

	//Classifier newClassifier = m_Classifier;
	Classifier newClassifier = copiesOfClassifier[0];
	newClassifier.buildClassifier(divData);
	m_Committee.add(newClassifier);
	double eComm = computeError(divData);//compute ensemble error
	if(m_Debug) System.out.println("Initialize:\tClassifier "+i+" added to ensemble. Ensemble error = "+eComm);
	
	//repeat till desired committee size is reached OR the max number of iterations is exceeded 
	while(i<m_DesiredSize && numTrials<m_NumIterations){
	    //Generate artificial training examples
	    artData = generateArtificialData(artSize, divData);
	    //Label artificial examples
	    labelData(artData);
	    addInstances(divData, artData);//Add new artificial data
	    
	    //Build new classifier
	    newClassifier = copiesOfClassifier[i]; 
	    newClassifier.buildClassifier(divData);
	    //Remove all the artificial data
	    removeInstances(divData, artSize);
	    
	    //Test if the new classifier should be added to the ensemble
	    m_Committee.add(newClassifier);//add new classifier to current committee
	    double currError = computeError(divData);
	    if(currError <= eComm){//adding the new member did not increase the error
		i++;
		eComm = currError;
		if(m_Debug) System.out.println("Iteration: "+(1+numTrials)+"\tClassifier "+i+" added to ensemble. Ensemble error = "+eComm);
	    }else{//reject the current classifier because it increased the ensemble error 
		m_Committee.removeElementAt(m_Committee.size()-1);//pop the last member
	    }
	    numTrials++;
	}
    }
    
    /** 
     * Compute and store statistics required for generating artificial data.
     *
     * @param data training instances
     * @exception Exception if statistics could not be calculated successfully
     */
    protected void computeStats(Instances data) throws Exception{
	int numAttributes = data.numAttributes();
	m_AttributeStats = new Vector(numAttributes);//use to map attributes to their stats
	
	for(int j=0; j<numAttributes; j++){
	    if(data.attribute(j).isNominal()){
		//Compute the probability of occurence of each distinct value 
		int []nomCounts = (data.attributeStats(j)).nominalCounts;
		double []counts = new double[nomCounts.length];
		if(counts.length < 2) throw new Exception("Nominal attribute has less than two distinct values!"); 
		//Perform Laplace smoothing
		for(int i=0; i<counts.length; i++)
		    counts[i] = nomCounts[i] + 1;
		Utils.normalize(counts);
		double []stats = new double[counts.length - 1];
		stats[0] = counts[0];
		//Calculate cumulative probabilities
		for(int i=1; i<stats.length; i++)
		    stats[i] = stats[i-1] + counts[i];
		m_AttributeStats.add(j,stats);
	    }else if(data.attribute(j).isNumeric()){
		//Get mean and standard deviation from the training data
		double []stats = new double[2];
		stats[0] = data.meanOrMode(j);
		stats[1] = Math.sqrt(data.variance(j));
		m_AttributeStats.add(j,stats);
	    }else System.err.println("Crate can only handle numeric and nominal values.");
	}
    }

    /**
     * Generate artificial training examples.
     * @param artSize size of examples set to create
     * @param data training data
     * @return the set of unlabeled artificial examples
     */
    protected Instances generateArtificialData(int artSize, Instances data){
	int numAttributes = data.numAttributes();
	Instances artData = new Instances(data, artSize);
	double []att; 
	Instance artInstance;
	
	for(int i=0; i<artSize; i++){
	    att = new double[numAttributes];
	    for(int j=0; j<numAttributes; j++){
		if(data.attribute(j).isNominal()){
		    //Select nominal value based on the frequency of occurence in the training data  
		    double []stats = (double [])m_AttributeStats.get(j);
		    att[j] =  (double) selectIndexProbabilistically(stats);
		}
		else if(data.attribute(j).isNumeric()){
		    //Generate numeric value from the Guassian distribution 
		    //defined by the mean and std dev of the attribute
		    double []stats = (double [])m_AttributeStats.get(j);
		    att[j] = (m_Random.nextGaussian()*stats[1])+stats[0];
		}else System.err.println("Crate can only handle numeric and nominal values.");
	    }
	    artInstance = new Instance(1.0, att);
	    artData.add(artInstance);
	}
	return artData;
    }

    /** 
     * Given cumulative probabilities select a nominal attribute value index 
     *
     * @param cdf array of cumulative probabilities
     * @return index of attribute selected based on the probability distribution 
     */
    protected int selectIndexProbabilistically(double []cdf){
	double rnd = m_Random.nextDouble();
	int index = 0;
	while(index < cdf.length && rnd > cdf[index]){
	    index++;
	}
	return index;
    }    
    
    /** 
     * Labels the artificially generated data.
     *
     * @param artData the artificially generated instances
     * @exception Exception if instances cannot be labeled successfully 
     */
    protected void labelData(Instances artData) throws Exception {
	Instance curr;
	double []preds = new double[m_Committee.size()];
	double mean,stdDev;
	
	for(int i=0; i<artData.numInstances(); i++){
	    curr = artData.instance(i);
	    //find the mean and std dev of predictions of committee members
	    for(int j=0; j<m_Committee.size(); j++)
		preds[j] = ((Classifier)m_Committee.get(j)).classifyInstance(curr);
	    
	    mean = Utils.mean(preds);
	    stdDev = Math.sqrt(Utils.variance(preds));
	    //select target value to be perturbed from the mean of the current committee's prediction by the alpha factor
	    //curr.setClassValue(m_Random.nextGaussian()*mean + m_Alpha*stdDev); 
	    if(m_Random.nextDouble()>0.5) 
	    	curr.setClassValue(mean + m_Alpha*stdDev); 
	    else curr.setClassValue(mean - m_Alpha*stdDev); 
	}	
    }
    
    /**
     * Removes a specified number of instances from the given set of instances.
     *
     * @param data given instances
     * @param numRemove number of instances to delete from the given instances
     */
    protected void removeInstances(Instances data, int numRemove){
	int num = data.numInstances();
	for(int i=num - 1; i>num - 1 - numRemove;i--){
	    data.delete(i);
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
    
    /** 
     * Computes the error in prediction on the given data.
     *
     * @param data the instances to be classified
     * @return mean absolute error
     * @exception Exception if error can not be computed successfully
     */
    protected double computeError(Instances data) throws Exception {
	double error;
	m_Evaluation = new Evaluation(data); //reset the counter in Evaluation
	m_Evaluation.evaluateModel(this, data);
	
	switch(m_ErrorMeasure){
	case MAE:
	    error = m_Evaluation.meanAbsoluteError();
	    break;
	case RMS:
	    error = m_Evaluation.rootMeanSquaredError();
	    break;
	case ROOT_RELATIVE_SQUARED:
	    error = m_Evaluation.rootRelativeSquaredError();
	    break;
	default:
	    error = m_Evaluation.meanAbsoluteError();
	}
	 
	return error;
    }
   
  /**
   * Classifies a given instance.
   *
   * @param instance the instance to be classified
   * @return the predicted value
   * @exception Exception if instance could not be predicted successfully
   */
    public double classifyInstance(Instance instance) throws Exception{
	if (!instance.classAttribute().isNumeric())
	    throw new UnsupportedClassTypeException("Crate is for numeric classes!");
	double pred = 0.0;
	Classifier curr;
	for (int i = 0; i < m_Committee.size(); i++) {
	    curr = (Classifier) m_Committee.get(i);
	    pred += curr.classifyInstance(instance);
	}
	pred /= m_Committee.size();
	return pred;
    }
    
    /**
     * Returns description of the Crate classifier.
     *
     * @return description of the Crate classifier as a string
     */
    public String toString() {
	
	if (m_Committee == null) {
	    return "Crate: No model built yet.";
	}
	StringBuffer text = new StringBuffer();
	text.append("Crate base classifiers: \n\n");
	for (int i = 0; i < m_Committee.size(); i++)
	    text.append(((Classifier) m_Committee.get(i)).toString() + "\n\n");
	text.append("Number of classifier in the ensemble: "+m_Committee.size()+"\n");
	return text.toString();
    }
    
    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String [] argv) {
	
	try {
	    System.out.println(Evaluation.evaluateModel(new Crate(), argv));
	} catch (Exception e) {
	    System.err.println(e.getMessage());
	}
    }
}
    
