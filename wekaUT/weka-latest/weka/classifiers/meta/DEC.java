/*
 *    DEC.java (Diverse Ensemble Classifier)
 *    Copyright (C) 2002 Prem Melville
 *
 *    UNDER DEVELOPMENT
 */

package weka.classifiers.meta;
import weka.classifiers.*;
import com.jmage.*;
import java.util.*;
import weka.core.*;
import weka.experiment.*;
/*
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Option;
import weka.core.Utils;
*/


/**
 * Class for creating Diverse Ensembles of a Classifier
 *
 * Valid options are:<p>
 *
 * -W classname <br>
 * Specify the full class name of a weak as the basis for 
 * DEC (required).<p>
 *
 * -I num <br>
 * Set the number of DEC iterations (default 50). <p>
 *
 * -N num <br>
 * Specify the desired size of the committee (default 15). <p>
 *
 * -S seed <br>
 * Random number seed for generating random examples (default random). <p>
 *
 * -R num <br>
 * Number of random instances to add at each iteration (default 20). <p>
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Prem Melville (melville@cs.utexas.edu)
 * @version 1.0
 */
public class DEC extends EnsembleClassifier
  implements OptionHandler{
        
    /** Use weights for committe votes - default equal wts*/
    protected int m_UseWeights = 0;
    
    /** The model base classifier to use */
    protected Classifier m_Classifier = new weka.classifiers.trees.j48.J48();
      
    /** Vector of classifiers that make up the committee */
    Vector committee=null;
    
    /** The number of iterations. */
    protected int m_NumIterations = 50;
    
    /** The number of iterations. */
    protected int m_DesiredSize = 15;
    
    /** The seed for random number generation. */
    protected int m_Seed = 0;
    
    /** Number of random instances to add at each iteration. */
    protected double m_RandomSize = 20.0 ;

    /** Confidence threshold above committee decisions are to be trusted. */
    protected double m_Threshold = 1.0 ;
    
    /** Possible methods to use for labeling randomly generated instances. */
    int LOW_PROB = 0,
	HIGH_PROB = 1,
	LEAST_LIKELY = 2,
	MOST_LIKELY = 3;
    
    /** Possible methods for creation of artificial data */
    int UNIFORM = 0,
	TRAINING_DIST = 1,
	MIXED = 2;
    
    /** Method to use for creation of artificial data */
    protected int m_DataCreationMethod = TRAINING_DIST;
    
    /** Method to use for labeling randomly generated instances. */
    protected int labeling_method = LOW_PROB;
    
    /** Random number generator */
    Random random = new Random(0);
    
    /** Attribute statistics */
    HashMap attribute_stats;
    
   /**
   * Returns an enumeration describing the available options
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(7);

    newVector.addElement(new Option(
	      "\tNumber of DEC iterations.\n"
	      + "\t(default 50)",
	      "I", 1, "-I <num>"));
    newVector.addElement(new Option(
	      "\tFull name of base classifier.\n"
	      + "\teg: weka.classifiers.NaiveBayes",
	      "W", 1, "-W"));
    newVector.addElement(new Option(
              "\tSeed for random number generator.\n"
              + "\t(default 0)",
              "S", 1, "-S"));
    newVector.addElement(new Option(
              "\rDesired size of committee.\n" 
              + "\t(default 15)",
              "N", 1, "-N"));
    newVector.addElement(new Option(
				    "\tConfidence threshold above committee decisions are to be trusted.\n"
				    + "\t(default 1.0)",
				    "C", 1, "-C"));
    
    newVector.addElement(new Option(
				    "\tNumber of random instances to add at each iteration.\n R=-1 uses the training set size.\n" 
				    + "\t(default 20)",
				    "R", 1, "-R"));
    
    newVector.addElement(new Option(				    
				    "\tMethod to use for artificial data generation (0=Uniform, 1=Training Distribution, 2=Mixed.\n"
				    + "\t(default 1)",
				    "A", 1, "-A"));

    newVector.addElement(new Option(
              "\tUse weights for committee votes (0=no weights, 1=proportional to accuracy).\n"
              + "\t(default 0)",
              "V", 1, "-V"));
    
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
   * Set the number of bagging iterations (default 50). <p>
   *
   * -S seed <br>
   * Random number seed for resampling (default 0).<p>
   *
   * -N num <br>
   * Specify the desired size of the committee (default 15). <p>
   *
   * -R num <br>
   * Number of random instances to add at each iteration (default 5). <p>
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    String bagIterations = Utils.getOption('I', options);
    if (bagIterations.length() != 0) {
      setNumIterations(Integer.parseInt(bagIterations));
    } else {
	setNumIterations(50);
    }

    String seed = Utils.getOption('S', options);
    if (seed.length() != 0) {
	setSeed(Integer.parseInt(seed));
    } else {
	setSeed(0);
    }

    String desired_size = Utils.getOption('N', options);
    if (desired_size.length() != 0) {
      setDesiredSize(Integer.parseInt(desired_size));
    } else {
	setDesiredSize(15);
    }

    String rnd_size = Utils.getOption('R', options);
    if (rnd_size.length() != 0) {
	setRandomSize(Double.parseDouble(rnd_size));
    } else {
	setRandomSize(20);
    }
    
    String threshold_str = Utils.getOption('C', options);
    if (threshold_str.length() != 0) {
	setThreshold(Double.parseDouble(threshold_str));
    } else {
	setThreshold(1.0);
    }
    //}catch (Exception e) { e.printStackTrace();};
    
    String data_create_str = Utils.getOption('A', options);
    if (data_create_str.length() != 0) {
	setDataCreationMethod(Integer.parseInt(data_create_str));
    } else {
	setDataCreationMethod(TRAINING_DIST);
    }
    
    String use_weights_str = Utils.getOption('V', options);
    if(use_weights_str.length() != 0) {
	setUseWeights(Integer.parseInt(use_weights_str));
    } else {
	setUseWeights(0);
    }

    String classifierName = Utils.getOption('W', options);
    if (classifierName.length() == 0) {
	throw new Exception("A classifier must be specified with"
			    + " the -W option.");
	//classifierName = default_classifier_name;
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
    String [] options = new String [classifierOptions.length + 17];
    int current = 0;
    options[current++] = "-S"; options[current++] = "" + getSeed();
    options[current++] = "-I"; options[current++] = "" + getNumIterations();
    options[current++] = "-N"; options[current++] = "" + getDesiredSize();
    options[current++] = "-R"; options[current++] = "" + getRandomSize();
    options[current++] = "-C"; options[current++] = "" + getThreshold();
    options[current++] = "-A"; options[current++] = "" + getDataCreationMethod();
    options[current++] = "-V"; options[current++] = "" + getUseWeights();
    
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
     * Set flag for using weights for committee votes.
     *
     * @param newUseWeights flag for using weights for committee votes.
     */
    public void setUseWeights(int newUseWeights){
	m_UseWeights = newUseWeights;
    }
    
    /**
     * Get flag for using weights for committee votes.
     *
     * @return flag for using weights for committee votes
     */
    public int getUseWeights(){
	return m_UseWeights;
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
   * Method to use for creating artificial data
   *
   * @return Method to use for creating artificial data
   */
    public int getDataCreationMethod() {

	return m_DataCreationMethod;
    }
  
  /**
   * Sets method to use for creating artificial data
   *
   * @param method the method to use for creating artificial data
   */
  public void setDataCreationMethod(int method) {
      
      m_DataCreationMethod = method;
  }


  /**
   * Number of random instances to add at each iteration.
   *
   * @return Number of random instances to add at each iteration
   */
    public double getRandomSize() {

	return m_RandomSize;
    }
  
  /**
   * Sets number of random instances to add at each iteration.
   *
   * @param new_random_size the number of random instances to add at each iteration.
   */
  public void setRandomSize(double new_random_size) {
      
      m_RandomSize = new_random_size;
  }
  
  /**
   * Gets the desired size of the committee.
   *
   * @return the bag size, as a percentage.
   */
  public int getDesiredSize() {

    return m_DesiredSize;
  }
  
  /**
   * Sets the desired size of the committee.
   *
   * @param newdesired_size the bag size, as a percentage.
   */
  public void setDesiredSize(int new_desired_size) {

    m_DesiredSize = new_desired_size;
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
	if(m_Seed==-1){
	    random = new Random();
	}else{
	    random = new Random(m_Seed);
	}
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
     * Get the value of threshold.
     * @return value of threshold.
     */
    public double getThreshold() {
	return m_Threshold;
    }
    
    /**
     * Set the value of threshold.
     * @param v  Value to assign to threshold.
     */
    public void setThreshold(double  v) {
	this.m_Threshold = v;
    }
    
    /**
     * DEC method.
     *
     * @param data the training data to be used for generating the
     * bagged classifier.
     * @exception Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
	
	//DEUBG
	if(m_UseWeights==1)
	    System.out.println("\t>> Using weights...");
	else
	    System.out.println("\t>> Not using weights...");
	
	//initialize ensemble wts to be equal 
	m_EnsembleWts = new double [m_DesiredSize];
	for(int j=0; j<m_DesiredSize; j++)
	    m_EnsembleWts[j] = 1.0;
	
	initMeasures();
	
	
	if (m_Classifier == null) {
	    throw new Exception("A base classifier has not been specified!");
	}
	if (data.checkForStringAttributes()) {
	    throw new Exception("Can't handle string attributes!");
	}
	
	//number of random instances to add at each iteration
	int random_size;
	if(m_RandomSize<0){
	    random_size = (int) (Math.abs(m_RandomSize)*data.numInstances());
	    if(random_size==0) random_size=1;//atleast add one random example
	} 
	else
	    random_size = (int) m_RandomSize;
	System.out.println("random size = "+random_size);
	
	//maximum number if random instances to generate (when using
	//confidence thresholds not all randomly generated examples
	//will be labeled).
	int max_random_generated = 3*random_size;
	int num_attributes = data.numAttributes();
	committee = new Vector();//initialize new committee
	double e_comm; //classification error of current committee
	int i = 1;//current size of committee
	int num_trials = 0;
	Classifier classifier = m_Classifier;
	Instances div_data = new Instances(data); //Local copy of data
	Instances random_data = null;
	computeStats(div_data);//Find mean and std devs for numeric data

	//Create first committee member - base
	m_Classifier.buildClassifier(div_data);
	committee.add(classifier);
	
	if(m_UseWeights==1)
	    m_EnsembleWts[i-1] = computeEnsembleWt(classifier, data);//compute wt based on classifier accuracy 
	
	e_comm = computeError(div_data);
	System.out.println("Mem "+i+" added. E_comm = "+e_comm);
	computeAccuracy(data);
	
	if(e_comm >= 0.5) {//if the base classifier has an error > 0.5
	    m_EnsembleWts[0] = 1.0;//reset the ensemble wt
	    i=m_DesiredSize; //skip the following while loop
	}
	
	while(i<m_DesiredSize && num_trials<m_NumIterations){
	    //System.out.println("\tTrial: "+num_trials);
	    
	    //Keeping generating random data until the desired number are actually labaled
	    Instances total_random_data = new Instances(data, random_size);
	    int labeled = 0, generated = 0;
	    
	    //Set confidence threshold for relabeling 
	    double threshold = selectThreshold(e_comm);
	    //System.out.println("\tThreshold: "+threshold);
	    
	    //Continue generating random examples until random_size of
	    //them are labeled or the the number of examples generated
	    //exceeds max_number_generated. The latter is a failsafe
	    //to ensure this loop terminates
	    while(labeled < random_size && generated < max_random_generated){
		//System.out.println("\tGenerating ramdom instances...");
		//Create random instances (diversity data)
		//random_data = generateRandomData(random_size, num_attributes, data);
		
		random_data = generateRandomData(random_size-labeled, num_attributes, data);
		generated += (random_size-labeled);
		
		//System.out.println("\tLabeling random instances...");
		//Label the random data
		random_data = labelData(random_data, threshold);
		
		labeled += random_data.numInstances();
		
		addInstances(total_random_data, random_data);
	    }
	    if(labeled!=random_size)
		System.out.println(labeled+" random examples labeled out of the desired "+random_size);
	    
	    Assert.that(total_random_data.numInstances()==labeled,"Error in random example generation+labeling loop: "+total_random_data.numInstances()+" != "+labeled);
	    
	    //Remove all the diversity data from the previous step (if any)
	    if(div_data.numInstances() > data.numInstances()) {
		//System.out.println("\tRemoving previous random data...");
		//removeInstances(div_data, random_size);
		removeInstances(div_data, div_data.numInstances()-data.numInstances());
	    }
	    
	    Assert.that(div_data.numInstances() == data.numInstances());
	    
	    //System.out.println("\tAdding new random instances...");
	    //Add new random data
	    addInstances(div_data, total_random_data);
	    
	    //System.out.println("\tBuild new classifier...");
	    //initialize new classifier
	    Classifier tmp[] = Classifier.makeCopies(m_Classifier,1);
	    classifier = tmp[0]; 
	    classifier.buildClassifier(div_data);
	    
	    committee.add(classifier);
	    if(m_UseWeights==1)
		m_EnsembleWts[i] = computeEnsembleWt(classifier, data);//compute wt based on classifier accuracy 
	    
	    //System.out.println("\tCompute current committee error...");
	    double curr_error = computeError(data);
	    if(m_EnsembleWts[i]>0 && curr_error <= e_comm){
		//adding the new member did not increase the error and the new member has an error < 0.5
		i++;
		e_comm = curr_error;
		System.out.println("Iteration: "+num_trials+"\tMem "+i+" added. E_comm = "+e_comm);
	    }else{
		committee.removeElementAt(committee.size()-1);//pop the last member
	    }
	    num_trials++;
	}
	System.out.println("Final ensemble size: "+committee.size());
	
	//Set measures
	computeEnsembleMeasures(data);
	//DEBUG
	Assert.that(m_TrainError == (100.0 * e_comm), "Bug in train error computation!"+m_TrainError+"\t"+(100.0 * e_comm));
}
    
    

    /** 
     * Find and store mean and std devs for numeric attributes.
     *
     * @param data training instances
     */
    protected void computeStats(Instances data){
	//Use to maintain the mean and std devs of numeric attributes
	int num_attributes = data.numAttributes();
	attribute_stats = new HashMap(num_attributes);
	
	for(int j=0; j<num_attributes; j++){
	    if(data.attribute(j).isNominal()){
		if(m_DataCreationMethod == TRAINING_DIST){
		    int []nom_counts = (data.attributeStats(j)).nominalCounts;
		    double []counts = new double[nom_counts.length];
		    //Laplace smoothing
		    for(int i=0; i<counts.length; i++)
			counts[i] = nom_counts[i] + 1;
		    Utils.normalize(counts);
		    double []stats = new double[counts.length - 1];
		    stats[0] = counts[0];
		    //Calculate cummalitive probabilities
		    for(int i=1; i<stats.length; i++)
			stats[i] = stats[i-1] + counts[i];
		    
		    attribute_stats.put(new Integer(j),stats);
		}
	    }else if(data.attribute(j).isNumeric()){
		if(m_DataCreationMethod == UNIFORM){
		    //get range of numeric attribute from the training data
		    Stats s = (data.attributeStats(j)).numericStats;
		    double []stats = new double[2];
		    stats[0] = s.min; 
		    stats[1] = s.max;
		    attribute_stats.put(new Integer(j),stats);
		}else if(m_DataCreationMethod == TRAINING_DIST){
		    //get mean and standard deviation from the training data
		    double []stats = new double[2];
		    stats[0] = data.meanOrMode(j);
		    stats[1] = Math.sqrt(data.variance(j));
		    attribute_stats.put(new Integer(j),stats);
		}
	    }
	}
    }
    
    protected Instances generateRandomData(int random_size, int num_attributes, Instances data){
	Instances random_data = new Instances(data, random_size);
	double []att; 
	Instance random_instance;
	
	for(int i=0; i<random_size; i++){
	    att = new double[num_attributes];
	    
	    for(int j=0; j<num_attributes; j++){
		if(data.attribute(j).isNominal()){
		    if(m_DataCreationMethod == UNIFORM || m_DataCreationMethod == MIXED){
			att[j] = (double) random.nextInt(data.numDistinctValues(j));
		    }else if(m_DataCreationMethod == TRAINING_DIST){
			double []stats = (double [])attribute_stats.get(new Integer(j));
			att[j] = selectNominalValue(stats);
		    }
		}else if(data.attribute(j).isNumeric()){
		    if(m_DataCreationMethod == UNIFORM){
			double []stats = (double [])attribute_stats.get(new Integer(j));
			double min = stats[0]; double max = stats[1];
			//System.out.println("<Min, Max> = "+min+"\t"+max);
			att[j] = (random.nextDouble() * (max - min)) + min;
		    }else if(m_DataCreationMethod == TRAINING_DIST || m_DataCreationMethod == MIXED){
			double []stats = (double [])attribute_stats.get(new Integer(j));
			att[j] = (random.nextGaussian()*stats[1])+stats[0];
			//System.out.println(data.attribute(j).name()+"\tMean= "+stats[0]+"\tStd Dev= "+stats[1]);
		    }			
		}else{
		    System.err.println("Current version of DEC cannot deal with STRING attributes.");
		}
		//System.out.println("\t Random att value: "+att[j]);
	    }
	    
	    random_instance = new Instance(1.0, att);
	    random_data.add(random_instance);
	}
	
	Assert.that(random_data.numInstances()==random_size);
	return random_data;
    }
    
    /** Given cummaltive probabilities select a nominal value index */
    protected double selectNominalValue(double []cumm){
	double rnd = random.nextDouble();
	int index = 0;
	while(index < cumm.length && rnd > cumm[index]){
	    index++;
	}
	return((double) index);
    }
    
    /**
     * Set threshold for relabeling based on user specified threhsold
     * or on error of current committee
     *
     * @param error Error of current committee
     * @return the selected threshold
     */
    protected double selectThreshold(double error){
	double threshold;
	
	if(m_Threshold == -1){
	    if(error >= 0.5) 
		threshold = 1.0;
	    else
		threshold = (error/(1.0 - error))*0.5 + 0.5;
	}else threshold=m_Threshold;
	
	return threshold;
    }
    
    /** 
     * Labels the randomly generated data.
     *
     * @param random_data the randomly generated instances
     * @param threhsold confidence threshold for relabeling data
     * @return labeled data
     * @exception Exception if instances cannot be labeled successfully */
    protected Instances labelData(Instances random_data, double threshold) throws Exception {
	Instances labeled = new Instances(random_data,1);
	Instance curr;
	double []probs;
	int ctr = 0;
	
	for(int i=0; i<random_data.numInstances(); i++){
	    curr = random_data.instance(i);
	    probs = distributionForInstance(curr);
	    
	    if(probs[Utils.maxIndex(probs)] <= threshold || committee.size()==1){
		ctr++;
		
	    if(labeling_method == LOW_PROB){
		curr.setClassValue(lowProbLabel(probs));
	    }else if(labeling_method == HIGH_PROB){
		curr.setClassValue(highProbLabel(probs));
	    }else if(labeling_method == LEAST_LIKELY){
		curr.setClassValue(Utils.minIndex(probs));//Assign the least likely label
	    }else if(labeling_method == MOST_LIKELY){
		curr.setClassValue(Utils.maxIndex(probs));//Assign the most likely label
	    }else{
		System.err.println("Unknown labeling method!");
	    }
	    
	    labeled.add(curr);
	    }
	}	
	return labeled;
    }
    
    /** 
     * Probabilisticly select class label - (high probability).
     *
     * @param probs posterior probability of each class
     * @return highly likely class label probabilistically selected
     */
    protected int highProbLabel(double []probs){
	double []cumm = new double[probs.length];
	
	//System.out.println("enter hi prob");
	//System.out.println("prob length = "+probs.length);

	//Compute cumulative probabilities 
	cumm[0] = probs[0];
	for(int i=1; i<probs.length; i++){
	    cumm[i] = probs[i]+cumm[i-1];
	}
	
	if(Double.isNaN(cumm[probs.length-1]))
	    System.err.println("Calculated cummaltive probability is NaN"); 
	//System.out.println("cumm = "+cumm[probs.length-1]);
	//Assert.that(Math.abs(cumm[probs.length-1] - 1)<0.00001,"Cummalative probability sums to "+cumm[probs.length-1]+" instead of 1.");
	//last value should be very close to one
	
	float rnd = random.nextFloat();
	int index = 0;
	while(rnd > cumm[index]){
	    index++;
	}
	
	//System.out.println("exit hi prob");

	return index;
    }
    
    
    /** 
     * Probabilisticly select class label - (low probability).
     *
     * @param probs posterior probability of each class
     * @return low probability class label probabilistically selected
     * @exception Exception if instances cannot be labeled successfully
     */
    protected int lowProbLabel(double []probs) throws Exception{
	double []inv_probs = new double[probs.length];
	
	//System.out.println("enter low prob");
	//System.out.println("prob length = "+probs.length);
	
	for(int i=0; i<probs.length; i++){
	    if(probs[i]==0){
		inv_probs[i] = Double.MAX_VALUE/probs.length; 
		//Hack to fix probability values of 0
		//Divide by probs.length to make sure normalizing works properly
	    }else{
		inv_probs[i] = 1.0 / probs[i];
	    }
	}
	
	Utils.normalize(inv_probs);
	
	//System.out.println("call hi prob");

	return highProbLabel(inv_probs);
    }
    
    /**
     *
     * @param div_data given instances
     * @param random_size number of instances to delete from the end of given instances
     */
    protected void removeInstances(Instances div_data, int random_size){
	int num = div_data.numInstances();
	for(int i=num - 1; i>num - 1 - random_size;i--){
	    div_data.delete(i);
	}
	
	Assert.that(div_data.numInstances() == num - random_size);
    }
    
    /**
     *
     * @param div_data given instances
     * @param random_data set of instances to add to given instances
     */
    protected void addInstances(Instances div_data, Instances random_data){
	for(int i=0; i<random_data.numInstances(); i++){
	    div_data.add(random_data.instance(i));
	}
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
	int num_instances = data.numInstances();
	Instance curr;
	
	for(int i=0; i<num_instances; i++){
	    curr = data.instance(i);
	    if(curr.classValue() != ((int) classifyInstance(curr))){//misclassified
		error++;
	    }
	}
	
	return (error/num_instances);
    }
    
    /** 
     * Compute ensemble weight.
     *
     * @param classifier current classifier
     * @param data instances to compute accuracy on 
     * @return computed vote weight for given classifier
     * @exception Exception if weight cannot be computed successfully
     */
    protected double computeEnsembleWt(Classifier classifier, Instances data) throws Exception{
	double wt = 0.0;
	//Compute error of classifier on data
	double error = 0.0;
	int num_instances = data.numInstances();
	Instance curr;
	double invBeta;
	
	for(int i=0; i<num_instances; i++){
	    curr = data.instance(i);
	    if(curr.classValue() != ((int) classifier.classifyInstance(curr))){//misclassified
		error++;
	    }
	}
	error = error/num_instances;
	if(error == 0.0)//prevent divide by zero error
	    invBeta = Double.MAX_VALUE;
	else
	    invBeta = ((1-error)/error);
	wt = Math.log(invBeta);
	return wt;
    }
    
    
     
     /** 
     * Computes classification accuracy on the given data.
     *
     * @param data the instances to be classified
     * @return classification accuracy
     * @exception Exception if error can not be computed successfully
     */
    protected double computeAccuracy(Instances data) throws Exception {
	double acc = 0.0;
	int num_instances = data.numInstances();
	Instance curr;
	
	for(int i=0; i<num_instances; i++){
	    curr = data.instance(i);
	    if(curr.classValue() == ((int) classifyInstance(curr))){//correctly classified
		acc++;
	    }
	}
	
	//System.out.println("# correctly classified: "+acc);
	//System.out.println("total #: "+num_instances);
	return (acc/num_instances);
    }
    
  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if distribution can't be computed successfully
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
      if(m_UseWeights==1){
	  return distributionForInstanceUsingWeights(instance);
      }else{
	  double [] sums = new double [instance.numClasses()], newProbs; 
	  
	  Classifier curr;
	  for (int i = 0; i < committee.size(); i++) {
	      curr = (Classifier) committee.get(i);
	      
	      if (instance.classAttribute().isNumeric() == true) {
		  sums[0] += curr.classifyInstance(instance);
	      } else if (curr instanceof DistributionClassifier) {
		  newProbs = ((DistributionClassifier)curr).distributionForInstance(instance);
		  for (int j = 0; j < newProbs.length; j++)
		      sums[j] += newProbs[j];
	      } else {
		  sums[(int)curr.classifyInstance(instance)]++;
	      }
	  }
	  if (instance.classAttribute().isNumeric() == true) {
	      sums[0] /= (double)(committee.size());
	      return sums;
	  } else if (Utils.eq(Utils.sum(sums), 0)) {
	      return sums;
	  } else {
	      Utils.normalize(sums);
	      return sums;
	  }
      }
  }

    /**
     * Calculates the class membership probabilities for the given test instance.
     * Incorporates vote weights.
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed successfully
     */
    public double[] distributionForInstanceUsingWeights(Instance instance) throws Exception {
	int commSize = committee.size();
	if (commSize == 0) {
	    throw new Exception("No model built");
	}
	double [] sums = new double [instance.numClasses()]; 
	Classifier curr;
	
	if (commSize == 1) {
	    curr = (Classifier) committee.get(0);
	    if (curr instanceof DistributionClassifier) {
		return ((DistributionClassifier)curr).distributionForInstance(instance);
	    } else {
		sums[(int)curr.classifyInstance(instance)] ++;
	    }
	} else {//commSize > 1
	    for (int i = 0; i < commSize; i++) {
		curr = (Classifier) committee.get(i);
		sums[(int)curr.classifyInstance(instance)] += m_EnsembleWts[i];
	    }
	}
	
	if (Utils.eq(Utils.sum(sums), 0)) {
	    return sums;
	} else {
	    Utils.normalize(sums);
	    return sums;
	}
    }
    
    /** Returns class predictions of each ensemble member */
    public double []getEnsemblePredictions(Instance instance) throws Exception{
	double preds[] = new double [committee.size()];
	for(int i=0; i<committee.size(); i++)
	    preds[i] = ((Classifier) committee.get(i)).classifyInstance(instance);
	
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
	return committee.size();
    }
    
   
  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  public String toString() {
    
      if (committee == null) {
	  return "DEC: No model built yet.";
      }
      StringBuffer text = new StringBuffer();
      text.append("All the base classifiers: \n\n");
      for (int i = 0; i < committee.size(); i++)
	  text.append(((Classifier) committee.get(i)).toString() + "\n\n");
      
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
			 evaluateModel(new DEC(), argv));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
}
