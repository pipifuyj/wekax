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
 *    SemiSupDecorate.java
 *    WARNING: UNDER DEVELOPMENT
 *    Copyright (C) 2004 Prem Melville
 *    
 */

package weka.classifiers.meta;
import weka.classifiers.*;
import java.util.*;
import weka.core.*;
import weka.experiment.*;

/**
 * SemiSupDecorate is an attempt to exploit unlabeled data to improve Decorate.
 *
 * DECORATE is a meta-learner for building diverse ensembles of
 * classifiers by using specially constructed artificial training
 * examples. Comprehensive experiments have demonstrated that this
 * technique is consistently more accurate than the base classifier, 
 * Bagging and Random Forests. Decorate also obtains higher accuracy than
 * Boosting on small training sets, and achieves comparable performance
 * on larger training sets.  For more
 * details see: <p>
 *
 * Prem Melville and Raymond J. Mooney. <i>Constructing diverse
 * classifier ensembles using artificial training examples.</i>
 * Proceedings of the Seventeeth International Joint Conference on
 * Artificial Intelligence 2003.<p>
 *
 * Prem Melville and Raymond J. Mooney. <i>Creating diversity in ensembles using artificial data.</i>
 * Journal of Information Fusion.<BR><BR>
 *
 * Valid options are:<p>
 *
 * -D <br>
 * Turn on debugging output.<p>
 *
 * -W classname <br>
 * Specify the full class name of a weak classifier as the basis for 
 * SemiSupDecorate (default weka.classifiers.trees.j48.J48()).<p>
 *
 * -I num <br>
 * Specify the desired size of the committee (default 15). <p>
 *
 * -M iterations <br>
 * Set the maximum number of SemiSupDecorate iterations (default 50). <p>
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
 * @version $Revision: 1.4 $ */
public class SemiSupDecorate extends EnsembleClassifier implements OptionHandler,SemiSupClassifier{

    /** Weight of unlabeled examples versus labeled examples */
    protected double m_Lambda = 1.0;
    
    /** Set of unlabeled examples */
    protected Instances m_Unlabeled;

    /** Set to true to use artificial data */
    protected boolean m_UseArtificial = true;
    
    /** Set to true to use unlabeled data */
    protected boolean m_UseUnlabeled = true; 
    //When set to false this should be equivalent to Decorate
    
    /** Types of unlabeled usage */
    static final int ALL = 0,
	IGNORE_LOW = 1,
	IGNORE_HIGH = 2,
	FLIP_LOW = 3;
    
    /** Type of usage of unlabeled examples */
    protected int m_UnlabeledMethod = ALL;
    
    /** Confidence threshold for labeling unlabeled examples */
    protected double m_Threshold = 0.9; 
    
    /** Set to true to get debugging output. */
    protected boolean m_Debug = false;

    /** The model base classifier to use. */
    protected Classifier m_Classifier = new weka.classifiers.trees.j48.J48();
      
    /** Vector of classifiers that make up the committee/ensemble. */
    protected Vector m_Committee = null;
    
    /** The desired ensemble size. */
    protected int m_DesiredSize = 15;

    /** The maximum number of SemiSupDecorate iterations to run. */
    protected int m_NumIterations = 50;
    
    /** The seed for random number generation. */
    protected int m_Seed = 0;
    
    /** Amount of artificial/random instances to use - specified as a
        fraction of the training data size. */
    protected double m_ArtSize = 1.0 ;

    /** The random number generator. */
    protected Random m_Random = new Random(0);
    
    /** Attribute statistics - used for generating artificial examples. */
    protected Vector m_AttributeStats = null;


    public void setLambda (double v) {
	m_Lambda = v;
    }

    public double getLambda () {
	return m_Lambda;
    }

    public String lambdaTipText() {
	return "set weight of unlabeled examples vs. labeled";
    }
    /**
     * Get the value of UseArtificial.
     * @return value of UseArtificial.
     */
    public boolean getUseArtificial() {
	return m_UseArtificial;
    }
    
    /**
     * Set the value of UseArtificial.
     * @param v  Value to assign to UseArtificial.
     */
    public void setUseArtificial(boolean  v) {
	m_UseArtificial = v;
    }
    
    /**
     * Get the value of Threshold.
     * @return value of Threshold.
     */
    public double getThreshold() {
	return m_Threshold;
    }
    
    /**
     * Set the value of Threshold.
     * @param v  Value to assign to Threshold.
     */
    public void setThreshold(double  v) {
	m_Threshold = v;
    }
    
    /**
     * Get the value of UnlabeledMethod.
     * @return value of UnlabeledMethod.
     */
    public int getUnlabeledMethod() {
	return m_UnlabeledMethod;
    }
    
    /**
     * Set the value of UnlabeledMethod.
     * @param v  Value to assign to UnlabeledMethod.
     */
    public void setUnlabeledMethod(int  v) {
	m_UnlabeledMethod = v;
    }
    
    /**
     * Get the value of UseUnlabeled.
     * @return value of UseUnlabeled.
     */
    public boolean getUseUnlabeled() {
	return m_UseUnlabeled;
    }
    
    /**
     * Set the value of UseUnlabeled.
     * @param v  Value to assign to UseUnlabeled.
     */
    public void setUseUnlabeled(boolean  v) {
	m_UseUnlabeled = v;
    }
    
    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    public Enumeration listOptions() {
	Vector newVector = new Vector(8);
	newVector.addElement(new Option(
	      "\tTurn on debugging output.",
	      "D", 0, "-D"));
	newVector.addElement(new Option(
              "\tDesired size of ensemble.\n" 
              + "\t(default 15)",
              "I", 1, "-I"));
	newVector.addElement(new Option(
	      "\tMaximum number of SemiSupDecorate iterations.\n"
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
				    "\tFactor that determines number of artificial examples to generate.\n"
				    +"\tSpecified proportional to training set size.\n" 
				    + "\t(default 1.0)",
				    "R", 1, "-R"));
    
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
     * SemiSupDecorate (required).<p>
     *
     * -I num <br>
     * Specify the desired size of the committee (default 15). <p>
     *
     * -M iterations <br>
     * Set the maximum number of SemiSupDecorate iterations (default 50). <p>
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
	setUseUnlabeled(Utils.getFlag('U', options));
	setUseArtificial(Utils.getFlag('A', options));
	setDebug(Utils.getFlag('D', options));

	String unlabeledMethod = Utils.getOption('Z', options);
	if (unlabeledMethod.length() != 0) {
	    setUnlabeledMethod(Integer.parseInt(unlabeledMethod));
	} else {
	    setUnlabeledMethod(0);
	}
	
	String threshold = Utils.getOption('T', options);
	if (threshold.length() != 0) {
	    setThreshold(Double.parseDouble(threshold));
	} else {
	    setThreshold(0.9);
	}
	
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

	String lambda = Utils.getOption('L', options);
	if (lambda.length() != 0) {
	    setLambda(Double.parseDouble(lambda));
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
	String [] options = new String [classifierOptions.length + 19];
	int current = 0;
	if (getDebug()) {
	    options[current++] = "-D";
	}
	if (getUseUnlabeled()) {
	    options[current++] = "-U";
	}
	if (getUseArtificial()) {
	    options[current++] = "-A";
	}
	options[current++] = "-T"; options[current++] = "" + getThreshold();
	options[current++] = "-Z"; options[current++] = "" + getUnlabeledMethod();
	options[current++] = "-S"; options[current++] = "" + getSeed();
	options[current++] = "-I"; options[current++] = "" + getDesiredSize();
	options[current++] = "-M"; options[current++] = "" + getNumIterations();
	options[current++] = "-R"; options[current++] = "" + getArtificialSize();
	options[current++] = "-L"; options[current++] = "" + getLambda();

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
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String desiredSizeTipText() {
      return "the desired number of member classifiers in the SemiSupDecorate ensemble. SemiSupDecorate may terminate "
	+"before this size is reached (depending on the value of numIterations). "
	+"Larger ensemble sizes usually lead to more accurate models, but increases "
	+"training time and model complexity.";
  }
    
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numIterationsTipText() {
    return "the maximum number of SemiSupDecorate iterations to run. Each iteration generates a classifier, "
	+"but does not necessarily add it to the ensemble. SemiSupDecorate stops when the desired ensemble "
	+"size is reached. This parameter should be greater than "
	+"equal to the desiredSize. If the desiredSize is not being reached it may help to "
	+"increase this value.";
  }
    
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String artificialSizeTipText() {
    return "determines the number of artificial examples to use during training. Specified as "
	+"a proportion of the training data. Higher values can increase ensemble diversity.";
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String seedTipText() {
    return "seed for random number generator used for creating artificial data."
	+" Set to -1 to use a random seed.";
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
b   */
  public String classifierTipText() {
    return "the desired base learner for the ensemble.";
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
      return "DECORATE is a meta-learner for building diverse ensembles of "
	  +"classifiers by using specially constructed artificial training "
	  +"examples. Comprehensive experiments have demonstrated that this "
	  +"technique is consistently more accurate than the base classifier, Bagging and Random Forests."
	  +"SemiSupDecorate also obtains higher accuracy than Boosting on small training sets, and achieves "
	  +"comparable performance on larger training sets. "
	  +"For more details see: P. Melville & R. J. Mooney. Constructing diverse classifier ensembles "
	  +"using artificial training examples (IJCAI 2003).\n"
	  +"P. Melville & R. J. Mooney. Creating diversity in ensembles using artificial data (submitted).";
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
     * Set the base classifier for SemiSupDecorate.
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
     * Sets the max number of SemiSupDecorate iterations to run.
     *
     * @param numIterations  max number of SemiSupDecorate iterations to run
     */
    public void setNumIterations(int numIterations) {
	m_NumIterations = numIterations;
    }

    /**
     * Gets the max number of SemiSupDecorate iterations to run.
     *
     * @return the  max number of SemiSupDecorate iterations to run
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
     * Provide unlabeled data to the classifier.
     * @unlabeled the unlabeled Instances
     */
    public void setUnlabeled(Instances unlabeled){
	m_Unlabeled = new Instances(unlabeled);//make local copy of unlabeled data
	//Reweight unlabeled instances if necessary
	if (m_Lambda != 1.0) 
	    weightInstances(m_Unlabeled, m_Lambda);
    }

    /** Weighted all given instances with given weight */
    protected void weightInstances (Instances insts, double weight) {
	Enumeration enumInsts = insts.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    instance.setWeight(weight);
	}
    }

    
    /**
     * Build SemiSupDecorate classifier
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
	if(data.classAttribute().isNumeric()) {
	    throw new UnsupportedClassTypeException("SemiSupDecorate can't handle a numeric class!");
	}
	if(m_NumIterations < m_DesiredSize)
	    throw new Exception("Max number of iterations must be >= desired ensemble size!");
	
	//initialize random number generator
	if(m_Seed==-1) m_Random = new Random();
	else m_Random = new Random(m_Seed);
	
	//initialize ensemble wts to be equal 
	m_EnsembleWts = new double [m_DesiredSize];
	for(int j=0; j<m_DesiredSize; j++)
	    m_EnsembleWts[j] = 1.0;
	initMeasures();
	
	int numUnlabeledUsed = 0;
	int i = 1;//current committee size
	int numTrials = 1;//number of SemiSupDecorate iterations 
	Instances divData = new Instances(data);//local copy of data - diversity data
	divData.deleteWithMissingClass();
	Instances artData = null;//artificial data

	//compute number of artficial instances to add at each iteration
	int artSize = (int) (Math.abs(m_ArtSize)*divData.numInstances());
	if(artSize==0) artSize=1;//atleast add one random example
	computeStats(data);//Compute training data stats for creating artificial examples
	
	//initialize new committee
	m_Committee = new Vector();
	Classifier newClassifier = m_Classifier;
	newClassifier.buildClassifier(divData);
	m_Committee.add(newClassifier);
	double eComm = computeError(divData);//compute ensemble error
	if(m_Debug) System.out.println("Initialize:\tClassifier "+i+" added to ensemble. Ensemble error = "+eComm);
	
	//repeat till desired committee size is reached OR the max number of iterations is exceeded 
	while(i<m_DesiredSize && numTrials<m_NumIterations){
	    
	    if(m_UseArtificial){
		//Generate artificial training examples
		artData = generateArtificialData(artSize, data);
		//Label artificial examples
		labelData(artData);
		//Add new artificial data
		addInstances(divData, artData);
	    }
	    
	    //Add unlabeled data
	    if(m_UseUnlabeled) numUnlabeledUsed = addUnlabeled(divData);
	    
	    //Build new classifier
	    Classifier tmp[] = Classifier.makeCopies(m_Classifier,1);
	    newClassifier = tmp[0]; 
	    newClassifier.buildClassifier(divData);
	    
	    //Remove unlabeled data
	    if(m_UseUnlabeled) removeInstances(divData, numUnlabeledUsed);
	    
	    //Remove all the artificial data
	    if(m_UseArtificial) removeInstances(divData, artSize);
	    
	    assert (divData.numInstances()==data.numInstances()) : "Diversity data error!";

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
     * Add unlabeled data to training set. 
     * @param divData diversity data to add to.
     * @return number of unlabeled examples used. 
     */
    protected int addUnlabeled(Instances divData) throws Exception{
	//set of unlabeled examples eventually used (some may be ignored). 
	Instances unlabeledUsed = new Instances(divData);
	switch (m_UnlabeledMethod){
	case ALL:
	    unlabeledUsed = labelAll(m_Unlabeled);
	    break;
	case IGNORE_LOW:
	    unlabeledUsed = labelIgnoreLow(m_Unlabeled, unlabeledUsed);
	    break;
	case IGNORE_HIGH:
	    unlabeledUsed = labelIgnoreHigh(m_Unlabeled, unlabeledUsed);
	    break;
	case FLIP_LOW:
	    unlabeledUsed = labelFlipLow(m_Unlabeled);
	    break;
	default:
	    System.err.println("Unrecognized unlabeled method selected!");
	}
	System.out.println("Unlabeled size = "+m_Unlabeled.numInstances()+" Used size = "+unlabeledUsed.numInstances());
	addInstances(divData, unlabeledUsed);
	return unlabeledUsed.numInstances();
    }
    
    /** Label examples as predicted by the current ensemble. */
    protected Instances labelAll(Instances instances) throws Exception {
	Instance curr;
	for(int i=0; i<instances.numInstances(); i++){
	    curr = instances.instance(i);
	    curr.setClassValue(classifyInstance(curr));
	}
	return m_Unlabeled;
    }

    /** 
     * Label high-confidence examples with the ensemble's prediction.
     * Ignore low-confidence examples.
     */
    protected Instances labelIgnoreLow(Instances instances, Instances used) throws Exception {
	Instance curr;
	double []probs;
	int label; 
	double highestProb;
	
	for(int i=0; i<instances.numInstances(); i++){
	    curr = instances.instance(i);
	    //compute the class membership probs predicted by the current ensemble 
	    probs = distributionForInstance(curr);
	    label = (int) classifyInstance(curr);
	    highestProb = probs[label];
	    
	    if(highestProb >= m_Threshold){
		curr.setClassValue(label);
		used.add(curr);
	    }	
	}
	return used;
    }
    
    /** 
     * Label low-confidence examples with inverse of ensemble's prediction.
     * Ignore high-confidence examples.
     */
    protected Instances labelIgnoreHigh(Instances instances, Instances used) throws Exception {
	Instance curr;
	double []probs;
	int label; 
	double highestProb;
	
	for(int i=0; i<instances.numInstances(); i++){
	    curr = instances.instance(i);
	    //compute the class membership probs predicted by the current ensemble 
	    probs = distributionForInstance(curr);
	    label = (int) classifyInstance(curr);
	    highestProb = probs[label];
	    
	    if(highestProb < m_Threshold){
		curr.setClassValue(inverseLabel(probs));
		used.add(curr);
	    }	
	}
	return used;
    }   
    
    /** 
     * Label low-confidence examples with inverse of ensemble's prediction.
     * Use ensemble's prediction for high-confidence examples.
     */
    protected Instances labelFlipLow(Instances instances) throws Exception {
	Instance curr;
	double []probs;
	int label; 
	double highestProb;
	
	int a=0, b=0;
	for(int i=0; i<instances.numInstances(); i++){
	    curr = instances.instance(i);
	    //compute the class membership probs predicted by the current ensemble 
	    probs = distributionForInstance(curr);
	    label = (int) classifyInstance(curr);
	    highestProb = probs[label];
	    
	    if(highestProb >= m_Threshold){
		curr.setClassValue(label);
		a++;
	    }else{
		curr.setClassValue(inverseLabel(probs));
		b++;
	    }
	}
	System.out.println("As is: "+a+"\tFlipped: "+b);
	return m_Unlabeled;
    }
    
    
    //Helper method to print arrays
    protected void printArray(double []array){
	for(int i=0; i<array.length; i++)
	    System.out.print(array[i]+" ");
	System.out.println();
    }
    
    /** Returns class predictions of each ensemble member */
    public double []getEnsemblePredictions(Instance instance) throws Exception{
	double preds[] = new double [m_Committee.size()];
	for(int i=0; i<m_Committee.size(); i++)
	    preds[i] = ((Classifier) m_Committee.get(i)).classifyInstance(instance);
	
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
	return m_Committee.size();
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
	    }else System.err.println("SemiSupDecorate can only handle numeric and nominal values.");
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
		}else System.err.println("SemiSupDecorate can only handle numeric and nominal values.");
	    }
	    artInstance = new Instance(1.0, att);
	    artData.add(artInstance);
	}
	return artData;
    }
    
    
    /** 
     * Labels the artificially generated data.
     *
     * @param artData the artificially generated instances
     * @exception Exception if instances cannot be labeled successfully 
     */
    protected void labelData(Instances artData) throws Exception {
	Instance curr;
	double []probs;
	
	for(int i=0; i<artData.numInstances(); i++){
	    curr = artData.instance(i);
	    //compute the class membership probs predicted by the current ensemble 
	    probs = distributionForInstance(curr);
	    //select class label inversely proportional to the ensemble predictions
	    curr.setClassValue(inverseLabel(probs));
	}	
    }
    

    /** 
     * Select class label such that the probability of selection is
     * inversely proportional to the ensemble's predictions.
     *
     * @param probs class membership probabilities of instance
     * @return index of class label selected
     * @exception Exception if instances cannot be labeled successfully 
     */
    protected int inverseLabel(double []probs) throws Exception{
	double []invProbs = new double[probs.length];
	//Produce probability distribution inversely proportional to the given
	for(int i=0; i<probs.length; i++){
	    if(probs[i]==0){
		invProbs[i] = Double.MAX_VALUE/probs.length; 
		//Account for probability values of 0 - to avoid divide-by-zero errors
		//Divide by probs.length to make sure normalizing works properly
	    }else{
		invProbs[i] = 1.0 / probs[i];
	    }
	}
	Utils.normalize(invProbs);
	double []cdf = new double[invProbs.length];
	//Compute cumulative probabilities 
	cdf[0] = invProbs[0];
	for(int i=1; i<invProbs.length; i++){
	    cdf[i] = invProbs[i]+cdf[i-1];
	}
	
	if(Double.isNaN(cdf[invProbs.length-1]))
	    System.err.println("Cumulative class membership probability is NaN!"); 
	return selectIndexProbabilistically(cdf);
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
     * Computes the error in classification on the given data.
     *
     * @param data the instances to be classified
     * @return classification error
     * @exception Exception if error can not be computed successfully
     */
    protected double computeError(Instances data) throws Exception {
	double error = 0.0;
	int numInstances = data.numInstances();
	Instance curr;
	
	for(int i=0; i<numInstances; i++){
	    curr = data.instance(i);
	    //Check if the instance has been misclassified
	    if(curr.classValue() != ((int) classifyInstance(curr))) error++;
	}
	return (error/numInstances);
    }
    
  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if distribution can't be computed successfully
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
      if (instance.classAttribute().isNumeric()) {
	  throw new UnsupportedClassTypeException("SemiSupDecorate can't handle a numeric class!");
      }
      double [] sums = new double [instance.numClasses()], newProbs; 
      Classifier curr;
      
      for (int i = 0; i < m_Committee.size(); i++) {
	  curr = (Classifier) m_Committee.get(i);
	  if (curr instanceof DistributionClassifier) {
	      newProbs = ((DistributionClassifier)curr).distributionForInstance(instance);
	      for (int j = 0; j < newProbs.length; j++)
		  sums[j] += newProbs[j];
	  } else {
	      sums[(int)curr.classifyInstance(instance)]++;
	  }
      }
      if (Utils.eq(Utils.sum(sums), 0)) {
	  return sums;
      } else {
	  Utils.normalize(sums);
	  return sums;
      }
  }
    
    /**
     * Returns description of the SemiSupDecorate classifier.
     *
     * @return description of the SemiSupDecorate classifier as a string
     */
    public String toString() {
	
	if (m_Committee == null) {
	    return "SemiSupDecorate: No model built yet.";
	}
	StringBuffer text = new StringBuffer();
	text.append("SemiSupDecorate base classifiers: \n\n");
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
	    System.out.println(Evaluation.evaluateModel(new SemiSupDecorate(), argv));
	} catch (Exception e) {
	    System.err.println(e.getMessage());
	}
    }
}
    
