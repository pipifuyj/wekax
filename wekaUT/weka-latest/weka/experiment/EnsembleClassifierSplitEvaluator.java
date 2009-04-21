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
 *    EnsembleClassifierSplitEvaluator.java
 *    Copyright (C) 2003 Prem Melville
 *
 */


package weka.experiment;

import java.io.*;
import java.util.*;

import weka.core.*;
import weka.classifiers.*;

/**
 * A SplitEvaluator that produces results for an ensemble classification scheme
 *
 * @author Prem Melville
 * @version $Revision: 1.3 $
 */
public class EnsembleClassifierSplitEvaluator 
    extends ClassifierSplitEvaluator implements SemiSupSplitEvaluator{ 
  
   /** The length of a result */
    private static final int RESULT_SIZE = 27;

    /** The number of IR statistics */
    private static final int NUM_IR_STATISTICS = 11;
    
    /** Class index for information retrieval statistics (default 0) */
    private int m_IRclass = 0;
    
  /**
   * Returns a string describing this split evaluator
   * @return a description of the split evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
      return " SplitEvaluator that produces results for an ensemble classification scheme ";
  }

  /**
   * Gets the data types of each of the result columns produced for a 
   * single run. The number of result fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array containing objects of the type of each result column. 
   * The objects should be Strings, or Doubles.
   */
  public Object [] getResultTypes() {
    int addm = (m_AdditionalMeasures != null) 
      ? m_AdditionalMeasures.length 
      : 0;
    int overall_length = RESULT_SIZE+addm;
    overall_length += NUM_IR_STATISTICS;
    Object [] resultTypes = new Object[overall_length];
    Double doub = new Double(0);
    int current = 0;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    //Ensemble stats - Prem Melville
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // IR stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Timing stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = "";

    // add any additional measures
    for (int i=0;i<addm;i++) {
      resultTypes[current++] = doub;
    }
    if (current != overall_length) {
      throw new Error("ResultTypes didn't fit RESULT_SIZE");
    }
    return resultTypes;
  }

  /**
   * Gets the names of each of the result columns produced for a single run.
   * The number of result fields must be constant
   * for a given SplitEvaluator.
   *
   * @return an array containing the name of each result column
   */
  public String [] getResultNames() {
    int addm = (m_AdditionalMeasures != null) 
      ? m_AdditionalMeasures.length 
      : 0;
    int overall_length = RESULT_SIZE+addm;
    overall_length += NUM_IR_STATISTICS;
    String [] resultNames = new String[overall_length];
    int current = 0;
    resultNames[current++] = "Number_of_instances";

    // Basic performance stats - right vs wrong
    resultNames[current++] = "Number_correct";
    resultNames[current++] = "Number_incorrect";
    resultNames[current++] = "Number_unclassified";
    resultNames[current++] = "Percent_correct";
    resultNames[current++] = "Percent_incorrect";
    resultNames[current++] = "Percent_unclassified";
    resultNames[current++] = "Kappa_statistic";

    //Ensemble stats - Prem Melville
    resultNames[current++] = "Ensemble_correct_mean_percent";
    resultNames[current++] = "Ensemble_incorrect_mean_percent";
    resultNames[current++] = "Ensemble_diversity";

    // Sensitive stats - certainty of predictions
    resultNames[current++] = "Mean_absolute_error";
    resultNames[current++] = "Root_mean_squared_error";
    resultNames[current++] = "Relative_absolute_error";
    resultNames[current++] = "Root_relative_squared_error";

    // SF stats
    resultNames[current++] = "SF_prior_entropy";
    resultNames[current++] = "SF_scheme_entropy";
    resultNames[current++] = "SF_entropy_gain";
    resultNames[current++] = "SF_mean_prior_entropy";
    resultNames[current++] = "SF_mean_scheme_entropy";
    resultNames[current++] = "SF_mean_entropy_gain";

    // K&B stats
    resultNames[current++] = "KB_information";
    resultNames[current++] = "KB_mean_information";
    resultNames[current++] = "KB_relative_information";

    // IR stats
    resultNames[current++] = "True_positive_rate";
    resultNames[current++] = "Num_true_positives";
    resultNames[current++] = "False_positive_rate";
    resultNames[current++] = "Num_false_positives";
    resultNames[current++] = "True_negative_rate";
    resultNames[current++] = "Num_true_negatives";
    resultNames[current++] = "False_negative_rate";
    resultNames[current++] = "Num_false_negatives";
    resultNames[current++] = "IR_precision";
    resultNames[current++] = "IR_recall";
    resultNames[current++] = "F_measure";

    // Timing stats
    resultNames[current++] = "Time_training";
    resultNames[current++] = "Time_testing";

    // Classifier defined extras
    resultNames[current++] = "Summary";
    // add any additional measures
    for (int i=0;i<addm;i++) {
      resultNames[current++] = m_AdditionalMeasures[i];
    }
    if (current != overall_length) {
      throw new Error("ResultNames didn't fit RESULT_SIZE");
    }
    return resultNames;
  }

  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param train the training Instances.
   * @param unlabeled the unlabled Instances.
   * @param test the testing Instances.
   * @return the results stored in an array. The objects stored in
   * the array may be Strings, Doubles, or null (for the missing value).
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances train, Instances unlabeled, Instances test) 
    throws Exception{
    
    if (train.classAttribute().type() != Attribute.NOMINAL) {
      throw new Exception("Class attribute is not nominal!");
    }
    if (m_Classifier == null) {
      throw new Exception("No classifier has been specified");
    }
    int addm = (m_AdditionalMeasures != null) 
      ? m_AdditionalMeasures.length 
      : 0;
    int overall_length = RESULT_SIZE+addm;
    overall_length += NUM_IR_STATISTICS;

    Object [] result = new Object[overall_length];
    EnsembleEvaluation eval = new EnsembleEvaluation(train);
    long trainTimeStart = System.currentTimeMillis();
    
    //Modification to allow for semisupervision
    if(m_Classifier instanceof SemiSupClassifier) ((SemiSupClassifier) m_Classifier).setUnlabeled(unlabeled);
    
    m_Classifier.buildClassifier(train);
    long trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
    long testTimeStart = System.currentTimeMillis();
    eval.evaluateModel(m_Classifier, test);
    long testTimeElapsed = System.currentTimeMillis() - testTimeStart;
    m_result = eval.toSummaryString();
    // The results stored are all per instance -- can be multiplied by the
    // number of instances to get absolute numbers
    int current = 0;
    result[current++] = new Double(eval.numInstances());
    result[current++] = new Double(eval.correct());
    result[current++] = new Double(eval.incorrect());
    result[current++] = new Double(eval.unclassified());
    result[current++] = new Double(eval.pctCorrect());
    result[current++] = new Double(eval.pctIncorrect());
    result[current++] = new Double(eval.pctUnclassified());
    result[current++] = new Double(eval.kappa());

    //Ensemble stats - Prem Melville
    result[current++] = new Double(eval.ensemblePctCorrect());
    result[current++] = new Double(eval.ensemblePctIncorrect());
    result[current++] = new Double(eval.ensembleDiversity());
    
    result[current++] = new Double(eval.meanAbsoluteError());
    result[current++] = new Double(eval.rootMeanSquaredError());
    result[current++] = new Double(eval.relativeAbsoluteError());
    result[current++] = new Double(eval.rootRelativeSquaredError());

    result[current++] = new Double(eval.SFPriorEntropy());
    result[current++] = new Double(eval.SFSchemeEntropy());
    result[current++] = new Double(eval.SFEntropyGain());
    result[current++] = new Double(eval.SFMeanPriorEntropy());
    result[current++] = new Double(eval.SFMeanSchemeEntropy());
    result[current++] = new Double(eval.SFMeanEntropyGain());

    // K&B stats
    result[current++] = new Double(eval.KBInformation());
    result[current++] = new Double(eval.KBMeanInformation());
    result[current++] = new Double(eval.KBRelativeInformation());

    // IR stats
    result[current++] = new Double(eval.truePositiveRate(m_IRclass));
    result[current++] = new Double(eval.numTruePositives(m_IRclass));
    result[current++] = new Double(eval.falsePositiveRate(m_IRclass));
    result[current++] = new Double(eval.numFalsePositives(m_IRclass));
    result[current++] = new Double(eval.trueNegativeRate(m_IRclass));
    result[current++] = new Double(eval.numTrueNegatives(m_IRclass));
    result[current++] = new Double(eval.falseNegativeRate(m_IRclass));
    result[current++] = new Double(eval.numFalseNegatives(m_IRclass));
    result[current++] = new Double(eval.precision(m_IRclass));
    result[current++] = new Double(eval.recall(m_IRclass));
    result[current++] = new Double(eval.fMeasure(m_IRclass));

    // Timing stats
    result[current++] = new Double(trainTimeElapsed / 1000.0);
    result[current++] = new Double(testTimeElapsed / 1000.0);

    if (m_Classifier instanceof Summarizable) {
      result[current++] = ((Summarizable)m_Classifier).toSummaryString();
    } else {
      result[current++] = null;
    }

    for (int i=0;i<addm;i++) {
      if (m_doesProduce[i]) {
	try {
	  double dv = ((AdditionalMeasureProducer)m_Classifier).
	    getMeasure(m_AdditionalMeasures[i]);
	  Double value = new Double(dv);

	  result[current++] = value;
	} catch (Exception ex) {
	  System.err.println(ex);
	}
      } else {
	result[current++] = null;
      }
    }

    if (current != overall_length) {
      throw new Error("Results didn't fit RESULT_SIZE");
    }
    return result;
  }

  /**
   * Gets the results for the supplied train and test datasets.
   *
   * @param train the training Instances.
   * @param test the testing Instances.
   * @return the results stored in an array. The objects stored in
   * the array may be Strings, Doubles, or null (for the missing value).
   * @exception Exception if a problem occurs while getting the results
   */
  public Object [] getResult(Instances train, Instances test) 
    throws Exception {
    
    if (train.classAttribute().type() != Attribute.NOMINAL) {
      throw new Exception("Class attribute is not nominal!");
    }
    if (m_Classifier == null) {
      throw new Exception("No classifier has been specified");
    }
    int addm = (m_AdditionalMeasures != null) 
      ? m_AdditionalMeasures.length 
      : 0;
    int overall_length = RESULT_SIZE+addm;
    overall_length += NUM_IR_STATISTICS;

    Object [] result = new Object[overall_length];
    EnsembleEvaluation eval = new EnsembleEvaluation(train);
    long trainTimeStart = System.currentTimeMillis();
    m_Classifier.buildClassifier(train);
    long trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
    long testTimeStart = System.currentTimeMillis();
    eval.evaluateModel(m_Classifier, test);
    long testTimeElapsed = System.currentTimeMillis() - testTimeStart;
    m_result = eval.toSummaryString();
    // The results stored are all per instance -- can be multiplied by the
    // number of instances to get absolute numbers
    int current = 0;
    result[current++] = new Double(eval.numInstances());
    result[current++] = new Double(eval.correct());
    result[current++] = new Double(eval.incorrect());
    result[current++] = new Double(eval.unclassified());
    result[current++] = new Double(eval.pctCorrect());
    result[current++] = new Double(eval.pctIncorrect());
    result[current++] = new Double(eval.pctUnclassified());
    result[current++] = new Double(eval.kappa());
    
    //Ensemble stats - Prem Melville
    result[current++] = new Double(eval.ensemblePctCorrect());
    result[current++] = new Double(eval.ensemblePctIncorrect());
    result[current++] = new Double(eval.ensembleDiversity());
    
    result[current++] = new Double(eval.meanAbsoluteError());
    result[current++] = new Double(eval.rootMeanSquaredError());
    result[current++] = new Double(eval.relativeAbsoluteError());
    result[current++] = new Double(eval.rootRelativeSquaredError());

    result[current++] = new Double(eval.SFPriorEntropy());
    result[current++] = new Double(eval.SFSchemeEntropy());
    result[current++] = new Double(eval.SFEntropyGain());
    result[current++] = new Double(eval.SFMeanPriorEntropy());
    result[current++] = new Double(eval.SFMeanSchemeEntropy());
    result[current++] = new Double(eval.SFMeanEntropyGain());

    // K&B stats
    result[current++] = new Double(eval.KBInformation());
    result[current++] = new Double(eval.KBMeanInformation());
    result[current++] = new Double(eval.KBRelativeInformation());

    // IR stats
    result[current++] = new Double(eval.truePositiveRate(m_IRclass));
    result[current++] = new Double(eval.numTruePositives(m_IRclass));
    result[current++] = new Double(eval.falsePositiveRate(m_IRclass));
    result[current++] = new Double(eval.numFalsePositives(m_IRclass));
    result[current++] = new Double(eval.trueNegativeRate(m_IRclass));
    result[current++] = new Double(eval.numTrueNegatives(m_IRclass));
    result[current++] = new Double(eval.falseNegativeRate(m_IRclass));
    result[current++] = new Double(eval.numFalseNegatives(m_IRclass));
    result[current++] = new Double(eval.precision(m_IRclass));
    result[current++] = new Double(eval.recall(m_IRclass));
    result[current++] = new Double(eval.fMeasure(m_IRclass));

    // Timing stats
    result[current++] = new Double(trainTimeElapsed / 1000.0);
    result[current++] = new Double(testTimeElapsed / 1000.0);

    if (m_Classifier instanceof Summarizable) {
      result[current++] = ((Summarizable)m_Classifier).toSummaryString();
    } else {
      result[current++] = null;
    }

    for (int i=0;i<addm;i++) {
      if (m_doesProduce[i]) {
	try {
	  double dv = ((AdditionalMeasureProducer)m_Classifier).
	    getMeasure(m_AdditionalMeasures[i]);
	  Double value = new Double(dv);

	  result[current++] = value;
	} catch (Exception ex) {
	  System.err.println(ex);
	}
      } else {
	result[current++] = null;
      }
    }

    if (current != overall_length) {
      throw new Error("Results didn't fit RESULT_SIZE");
    }
    return result;
  }


  /**
   * Returns a text description of the split evaluator.
   *
   * @return a text description of the split evaluator.
   */
  public String toString() {

    String result = "EnsembleClassifierSplitEvaluator: ";
    if (m_Classifier == null) {
      return result + "<null> classifier";
    }
    return result + m_Classifier.getClass().getName() + " " 
      + m_ClassifierOptions + "(version " + m_ClassifierVersion + ")";
  }
} // EnsembleClassifierSplitEvaluator







