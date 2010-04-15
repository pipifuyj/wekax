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

package weka.classifiers.bayes;

import weka.classifiers.*;
import weka.classifiers.sparse.*;
import java.io.*;
import java.util.*;
import weka.core.*;

/**
 * Semi supervised learner that uses EM initialized with labeled data and then
 * runs EM iterations on the unlabeled data to improve the model.
 *
 * See: Kamal Nigam, Andrew McCallum, Sebastian Thrun and Tom
 * Mitchell. Text Classification from Labeled and Unlabeled Documents
 * using EM. Machine Learning, 39(2/3). pp. 103-134. 2000.
 *
 * Assumes use of a base classifier that is a SoftClassifer that
 * accepts training data with a soft class distribution rather than
 * a hard assignment, i.e. SoftClassifiedInstances. Sample soft
 * classifiers are NaiveBayesSimpleSoft and NaiveBayesSimpleSparseSoft
 *
 * @author Ray Mooney  (mooney@cs.utexas.edu)*/

public class SemiSupEM extends DistributionClassifier implements SemiSupClassifier, OptionHandler{

    /** Original set of unlabeled Instances */
    protected Instances m_UnlabeledData;

    /** Soft labeled version of unlabeled data */
    protected SoftClassifiedInstances m_UnlabeledInstances;

    /** Hard Labeled data */
    protected Instances m_LabeledInstances;

    /** Complete set of labeled and unlabeled instances for EM */
    protected SoftClassifiedInstances m_AllInstances;

    /** Base classifier that supports soft classified instances */
    protected SoftClassifier m_Classifier = new NaiveBayesSimpleSoft();

    /** Weight of unlabeled examples during EM training versus labeled examples (see Nigam et al.)*/
    protected double m_Lambda = 1.0;

    /** random numbers and seed */
    protected Random m_Random;
    protected int m_rseed;
    
    /** maximum iterations to perform */
    protected int m_max_iterations;

    /** Create soft labeled Seed for unseen classes */
    protected boolean m_seedUnseenClasses;

    /** Verbose? */
    protected boolean m_verbose;

    protected static double m_minLogLikelihoodIncr = 1e-6;

    /** The minimum values for numeric attributes. */
    protected double [] m_MinArray;
    
    /** The maximum values for numeric attributes. */
    protected double [] m_MaxArray;

    /**
     * Returns a string describing this clusterer
     * @return a description of the evaluator suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
	return "Classifier trained using both labeled and unlabeled data using EM";
    }

    /**
     * Returns an enumeration describing the available options.. <p>
     *
     * Valid options are:<p>
     *
     * -V <br>
     * Verbose. <p>
     *
     * -I <max iterations> <br>
     * Terminate after this many iterations if EM has not converged. <p>
     *
     * -S <seed> <br>
     * Specify random number seed. <p>
     *
     * -M <num> <br>
     *  Set the minimum allowable standard deviation for normal density 
     * calculation. <p>
     *
     * @return an enumeration of all the available options.
     *
     **/
    public Enumeration listOptions () {
	Vector newVector = new Vector(7);
	newVector.addElement(new Option(
					"\tFull name of classifier to boost.\n"
					+"\teg: weka.classifiers.bayes.NaiveBayes",
					"W", 1, "-W <class name>"));
	newVector.addElement(new Option("\tLambda weight for unlabeled data.\n(default 1)", "L"
					, 1, "-L <num>"));
	newVector.addElement(new Option("\tmax iterations.\n(default 100)", "I"
					, 1, "-I <num>"));
	newVector.addElement(new Option("\trandom number seed.\n(default 1)"
					, "S", 1, "-S <num>"));
	newVector.addElement(new Option("\tverbose.", "V", 0, "-V"));
	newVector.addElement(new Option("\tSeed unseen classes.", "U", 0, "-U"));
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

	String classifierName = Utils.getOption('W', options);
	if (classifierName.length() == 0) {
	    throw new Exception("A classifier must be specified with"
				+ " the -W option.");
	}
	setClassifier((SoftClassifier)Classifier.forName(classifierName,
							 Utils.partitionOptions(options)));

	setDebug(Utils.getFlag('V', options));
	setSeedUnseenClasses(Utils.getFlag('U', options));

	String optionString = Utils.getOption('I', options);

	if (optionString.length() != 0) {
	    setMaxIterations(Integer.parseInt(optionString));
	}
	
	optionString = Utils.getOption('S', options);

	if (optionString.length() != 0) {
	    setSeed(Integer.parseInt(optionString));
	}

	optionString = Utils.getOption('L', options);

	if (optionString.length() != 0) {
	    setLambda(Double.parseDouble(optionString));
	}

    }

    /**
     * Reset to default options
     */
    protected void resetOptions () {
	m_max_iterations = 100;
	m_rseed = 100;
	m_verbose = false;
	m_seedUnseenClasses = false;
	m_Classifier = new NaiveBayesSimpleSoft();
	m_Lambda = 1.0;
    }


    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String seedTipText() {
	return "random number seed";
    }


    /**
     * Set the random number seed
     *
     * @param s the seed
     */
    public void setSeed (int s) {
	m_rseed = s;
    }


    /**
     * Get the random number seed
     *
     * @return the seed
     */
    public int getSeed () {
	return  m_rseed;
    }


    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String maxIterationsTipText() {
	return "maximum number of EM iterations";
    }

    /**
     * Set the maximum number of iterations to perform
     *
     * @param i the number of iterations
     * @exception Exception if i is less than 1
     */
    public void setMaxIterations (int i)
	throws Exception
    {
	if (i < 1) {
	    throw  new Exception("Maximum number of iterations must be > 0!");
	}

	m_max_iterations = i;
    }


    /**
     * Get the maximum number of iterations
     *
     * @return the number of iterations
     */
    public int getMaxIterations () {
	return  m_max_iterations;
    }


    /**
     * Set debug mode - verbose output
     *
     * @param v true for verbose output
     */
    public void setDebug (boolean v) {
	m_verbose = v;
    }


    /**
     * Get debug mode
     *
     * @return true if debug mode is set
     */
    public boolean getDebug () {
	return  m_verbose;
    }

    public void setSeedUnseenClasses (boolean v) {
	m_seedUnseenClasses = v;
    }

    public boolean getSeedUnseenClasses () {
	return m_seedUnseenClasses;
    }

    public String seedUnseenClassesTipText() {
	return "create soft seeds for unseen classes using farthest-first";
    }


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
     * Set the classifier for boosting. 
     *
     * @param newClassifier the Classifier to use.
     */
    public void setClassifier(SoftClassifier newClassifier) {

	m_Classifier = newClassifier;
    }

    /**
     * Get the classifier used as the classifier
     *
     * @return the classifier used as the classifier
     */
    public SoftClassifier getClassifier() {

	return m_Classifier;
    }

    public String classifierTipText() {
	return "Base SoftClassifier to use for underlying probabilistic classification";
    }


    /**
     * Gets the current settings of EM.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions () {
	String [] classifierOptions = new String [0];
	if ((m_Classifier != null) && 
	    (m_Classifier instanceof OptionHandler)) {
	    classifierOptions = ((OptionHandler)m_Classifier).getOptions();
	}
	
	String [] options = new String [classifierOptions.length + 10];
	int current = 0;
	if (m_verbose) {
	    options[current++] = "-V";
	}
	if (m_seedUnseenClasses) {
	    options[current++] = "-U";
	}
	options[current++] = "-I";
	options[current++] = "" + m_max_iterations;
	options[current++] = "-S";
	options[current++] = "" + m_rseed;
	options[current++] = "-L";
	options[current++] = "" + m_Lambda;

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

	return  options;
    }

    /** 
     * Provide unlabeled data to the classifier.
     * @unlabeled the unlabeled Instances
     */
    public void setUnlabeled(Instances unlabeled){
	m_UnlabeledData = unlabeled;
    }

    /** Simple constructor, must set options using command line or GUI */
    public SemiSupEM() {
	resetOptions();
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data 
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances data) throws Exception {
	if (data.checkForStringAttributes()) {
	    throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
	}
	if (data.classAttribute().isNumeric()) {
	    throw new UnsupportedClassTypeException("Can't handle a numeric class!");
	}
	if (m_Classifier == null) {
	    throw new Exception("A base classifier has not been specified!");
	}
	m_LabeledInstances = data;
	// Add "hard" soft-labeled instances of labeled data to the data for EM
	m_AllInstances = new SoftClassifiedInstances(data);
	Random m_Random = new Random(m_rseed);
	// Make random soft-labeled instances for unlabeled data
	m_UnlabeledInstances = new SoftClassifiedInstances(m_UnlabeledData, m_Random);
	if (m_Lambda != 1.0) 
	    weightInstances(m_UnlabeledInstances, m_Lambda);
	// Add the unlabeled data to the complete data set
	m_AllInstances.addInstances(m_UnlabeledInstances);
	initModel();
  	if (m_verbose) {
	    System.out.println("Labeled Data Classes: ");
	    Enumeration enumInsts = m_LabeledInstances.enumerateInstances();
	    while (enumInsts.hasMoreElements()) {
		Instance instance = (Instance) enumInsts.nextElement();
		System.out.print(m_AllInstances.classAttribute().value((int)instance.classValue()) + " ");
	    }
	    System.out.println("\nNum Unlabeled: " + m_UnlabeledInstances.numInstances() );
	    //  	    System.out.println("Labeled data: " + m_LabeledInstances);
	    //  	    System.out.println("Unlabeled data: " + m_UnlabeledInstances);
  	}
	if (m_UnlabeledInstances.numInstances() != 0)
	    iterate();
    }

    /** Weighted all given instances with given weight */
    protected void weightInstances (Instances insts, double weight) {
	Enumeration enumInsts = insts.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    instance.setWeight(weight);
	}
    }

    /** Intialize model using appropriate set of data */
    protected void initModel() throws Exception {
	SoftClassifiedInstances seedInstances = new SoftClassifiedInstances(m_LabeledInstances);
	if (m_seedUnseenClasses && m_UnlabeledInstances.numInstances() != 0) {
	    List unseenClasses = unseenClasses(seedInstances);
	    if (!unseenClasses.isEmpty()) {
		if (m_verbose) 
		    System.out.println("Unseen classes: " + unseenClasses);
		// Add a seed instance for the unseen classes that is soft labeled equally
		// in all unkown classes.
		Instance farthest =  farthestInstance(m_UnlabeledInstances, seedInstances);
		softLabelClasses((SoftClassifiedInstance)farthest, unseenClasses);
		if (m_verbose) 
		    System.out.println("Seeded Instance: " + classDistributionString((SoftClassifiedInstance)farthest));
		seedInstances.add(farthest);
	    }
	}
	m_Classifier.buildClassifier(seedInstances);
    }

    /** Return a list of class values for which there are no
     *  instances in insts */
    protected ArrayList unseenClasses(Instances insts) {
	int[] classCounts = new int[insts.numClasses()];
	Enumeration enumInsts = insts.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance inst = (Instance) enumInsts.nextElement();
	    classCounts[(int)inst.classValue()]++;
	}
	ArrayList result = new ArrayList();
	for (int i = 0; i < insts.numClasses(); i++) {
	    if (classCounts[i] == 0) {
		result.add(new Integer(i));
	    }
	}
	return result;
    }

    /** Return the instance in candidateInsts that is farthest from any instance
     * in insts */
    protected Instance farthestInstance(Instances candidateInsts, Instances insts) {
	double maxDist = Double.NEGATIVE_INFINITY;
	Instance farthestInst = null;
	double dist;
	setMinMax(m_AllInstances);
	Enumeration enumInsts = candidateInsts.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance candidate = (Instance) enumInsts.nextElement();
	    dist = minimumDistance(candidate, insts);
	    if (dist > maxDist) {
		maxDist = dist;
		farthestInst = candidate;
	    }
	}
	return farthestInst;
    }

    /** Return the distance from inst to the closest instance in insts */
    protected double minimumDistance(Instance inst, Instances insts) {
	double minDist = Double.POSITIVE_INFINITY;
	double dist;
	Enumeration enumInsts = insts.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance X = (Instance) enumInsts.nextElement();
	    dist = distance(inst, X);
	    if (dist < minDist) {
		minDist = dist;
	    }
	}
	return minDist;
    }

    /** Soft label inst as being equally likely to be in an of the given classes */	
    protected void softLabelClasses(SoftClassifiedInstance inst, List classes) 
	throws Exception {
	double prob = 1.0/classes.size();
	double[] dist = new double[((Instance)inst).dataset().numClasses()];
	for (int i = 0; i < classes.size(); i++) {
	    dist[((Integer)classes.get(i)).intValue()] = prob;
	}
	inst.setClassDistribution(dist);	
    }

    /** Run EM iterations until likelihood stops increasing significantly or max iterations exhausted */
    protected void iterate() throws Exception {
	double logLikelihood, oldLogLikelihood;
	logLikelihood = 0;
	oldLogLikelihood = 0;
	for (int i = 0; i < m_max_iterations; i++) {
	    //	     if (m_verbose) {
	    //	    		System.out.println(m_Classifier);
	    //	    	     }
	    oldLogLikelihood = logLikelihood;
	    logLikelihood = eStep();
	    if (m_verbose) {
		System.out.println("\nIteration " + i + ":  LogLikelihood = " + logLikelihood + "\n\n");
	    }
	    if ( (i > 0) && ((logLikelihood - oldLogLikelihood) < m_minLogLikelihoodIncr))
		break;
	    mStep();
	}
    }
	
    protected double eStep() throws Exception {
	double logLikelihood = 0;
	double classifiedCorrect = 0;
	double[] dist;
	Enumeration enumInsts = m_UnlabeledInstances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    dist = m_Classifier.unNormalizedDistributionForInstance(instance);
	    //	  instance.setClassDistribution(dist);
	    //    System.out.println("Instance:" + instance + " Dist: " + classDistributionString(instance));
	    logLikelihood += logSum(dist);
	    NaiveBayesSimple.normalizeLogs(dist);
	    //	    System.out.println("Norm Dist: " + classDistributionString((SoftClassifiedInstance)instance));
	    ((SoftClassifiedInstance)instance).setClassDistribution(dist);
	    if (m_verbose) {
		// System.out.println(classDistributionString(instance));
		if (Utils.maxIndex(dist) == (int)instance.classValue()) {
		    classifiedCorrect++;
		}
	    }
	}
	if (m_verbose) {
	    System.out.println("\nAccuracy on Unlabeled: " + classifiedCorrect/ m_UnlabeledInstances.numInstances());
	}
	enumInsts = m_LabeledInstances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    dist = m_Classifier.unNormalizedDistributionForInstance(instance);
	    logLikelihood += logSum(dist);
	}
	return logLikelihood/m_AllInstances.numInstances();
    }

    /** Sums log of probabilities using special method for summing in log space
     */
    public double logSum(double[] logProbs) {
	double sum = 0;
 	double max = logProbs[Utils.maxIndex(logProbs)];
	for (int i = 0; i < logProbs.length; i++) {
	    sum +=  Math.exp(logProbs[i] - max);
	}
	return max + Math.log(sum);
    }


    protected String classDistributionString(SoftClassifiedInstance inst) {
	double[] dist = inst.getClassDistribution();
	StringBuffer text = new StringBuffer();
	Attribute classAtt = m_AllInstances.classAttribute();
	text.append(classAtt.value((int)((Instance)inst).classValue()) + " | ");
	for (int i = 0; i < m_AllInstances.numClasses(); i++) {
	    text.append(classAtt.value(i) + ":" + dist[i] + " ");
	}
	return text.toString();
    }


    protected void mStep() throws Exception {
	m_Classifier.buildClassifier(m_AllInstances);
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
	double[] dist = m_Classifier.unNormalizedDistributionForInstance(instance);
	NaiveBayesSimple.normalizeLogs(dist);
	return dist;
    }

    /**
     * Calculates the distance between two instances
     *
     * @param first the first instance
     * @param second the second instance
     * @return the distance between the two given instances
     */          
    protected double distance(Instance first, Instance second) {
    
	double diff, distance = 0;
	Instances dataset = first.dataset();

	for(int i = 0; i < dataset.numAttributes(); i++) { 
	    if (i == dataset.classIndex()) {
		continue;
	    }
	    if (dataset.attribute(i).isNominal()) {

		// If attribute is nominal
		if (first.isMissing(i) || second.isMissing(i) ||
		    ((int)first.value(i) != (int)second.value(i))) {
		    distance += 1;
		}
	    } else {
	
		// If attribute is numeric
		if (first.isMissing(i) || second.isMissing(i)){
		    if (first.isMissing(i) && second.isMissing(i)) {
			diff = 1;
		    } else {
			if (second.isMissing(i)) {
			    diff = norm(first.value(i), i);
			} else {
			    diff = norm(second.value(i), i);
			}
			if (diff < 0.5) {
			    diff = 1.0 - diff;
			}
		    }
		} else {
		    diff = norm(first.value(i), i) - norm(second.value(i), i);
		}
		distance += diff * diff;
	    }
	}
    
	return distance;
    }
    
    /**
     * Normalizes a given value of a numeric attribute.
     *
     * @param x the value to be normalized
     * @param i the attribute's index
     */
    protected double norm(double x,int i) {

	if (Double.isNaN(m_MinArray[i])
	    || Utils.eq(m_MaxArray[i], m_MinArray[i])) {
	    return 0;
	} else {
	    return (x - m_MinArray[i]) / (m_MaxArray[i] - m_MinArray[i]);
	}
    }

    /** Compute and store min max values for each numeric feature */
    protected void setMinMax(Instances insts) {
	m_MinArray = new double [insts.numAttributes()];
	m_MaxArray = new double [insts.numAttributes()];
	for (int i = 0; i < insts.numAttributes(); i++) {
	    m_MinArray[i] = m_MaxArray[i] = Double.NaN;
	}
	Enumeration enum = insts.enumerateInstances();
	while (enum.hasMoreElements()) {
	    updateMinMax((Instance) enum.nextElement());
	}
    }

    /**
     * Updates the minimum and maximum values for all the attributes
     * based on a new instance.
     *
     * @param instance the new instance
     */
    protected void updateMinMax(Instance instance) {
	Instances dataset = instance.dataset();
    
	for (int j = 0;j < dataset.numAttributes(); j++) {
	    if ((dataset.attribute(j).isNumeric()) && (!instance.isMissing(j))) {
		if (Double.isNaN(m_MinArray[j])) {
		    m_MinArray[j] = instance.value(j);
		    m_MaxArray[j] = instance.value(j);
		} else {
		    if (instance.value(j) < m_MinArray[j]) {
			m_MinArray[j] = instance.value(j);
		    } else {
			if (instance.value(j) > m_MaxArray[j]) {
			    m_MaxArray[j] = instance.value(j);
			}
		    }
		}
	    }
	}
    }


    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    //    public static void main(String [] argv) {

    //      try {
    //        NaiveBayesSimpleSoft baseClassifier = new NaiveBayesSimpleSoft();
    //        baseClassifier.setMinStdDev(.15);
    //        Instances instances = new Instances(new BufferedReader(new FileReader(argv[0])));
    //        instances.setClassIndex(instances.numAttributes() - 1);
    //        SemiSupEM emClassifier = new SemiSupEM();
    //        emClassifier.resetOptions();
    //        emClassifier.setClassifier(baseClassifier);
    //        emClassifier.setDebug(true);
    //        //      emClassifier.setUnlabeledSeeding(true);
    //        Random random = new Random();
    //        instances.randomize(random);
    //        int numLabeled = Integer.parseInt(argv[1]);
    //        Instances labeledInsts = new Instances(instances, 0, numLabeled);
    //        Instances unlabeledInsts = new Instances(instances, numLabeled, (instances.numInstances() - numLabeled));
    //        emClassifier.setUnlabeled(unlabeledInsts);
    //        emClassifier.buildClassifier(labeledInsts);
    //      } catch (Exception e) {
    //        System.err.println(e.getMessage());
    //      }
    //    }

    public static void main(String [] argv) {
	try {
	    Instances instances = new Instances(new BufferedReader(new FileReader(argv[0])));
	    instances.setClassIndex(instances.numAttributes() - 1);
	    Random random = new Random(Integer.parseInt(argv[2]));
	    instances.randomize(random);
	    int numLabeled = Integer.parseInt(argv[1]);
	    Instances labeledInsts = new Instances(instances, 0, numLabeled);
	    Instances unlabeledInsts = new Instances(instances, numLabeled, (instances.numInstances() - numLabeled));
	    SemiSupEM emClassifier = new SemiSupEM();
	    emClassifier.setOptions(argv);
	    emClassifier.setUnlabeled(unlabeledInsts);
	    emClassifier.buildClassifier(labeledInsts);
	} catch (Exception e) {
	    System.err.println(e.getMessage());
	}
    }


}
