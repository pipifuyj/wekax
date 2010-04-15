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
 *    SemiSupClustererEvaluation.java
 *    Copyright (C) 2002 Sugato Basu, Misha Bilenko
 *
 */

package  weka.clusterers;

import  java.util.*;
import  java.io.*;
import  weka.core.*;
import  weka.filters.Filter;
import  weka.filters.unsupervised.attribute.Remove;

/**
 * Class for evaluating clustering models - extends ClusterEvaluation.java<p>
 * Implements different clustering evaluation metrics
 *
 * @author   Sugato Basu, Misha Bilenko
 */
public class SemiSupClustererEvaluation extends ClusterEvaluation {

  /** Purity of the clustering */
  protected double m_Purity;

  /** Entropy of the clustering */
  protected double m_Entropy;

  /** Objective function of the clustering */
  protected double m_Objective;

  /** MI Metric the clustering */
  protected double m_MIMetric;

  /** KL Divergence of the clustering */
  protected double m_KLDivergence;

  /** The number of underlying classes */
  protected int m_NumClasses;

  /** The number of produced clusters */
  protected int m_NumClusters;

  /** All labeled training instances */
  protected Instances m_LabeledTrain;

  /** All unlabaled training instances */
  protected Instances m_UnlabeledTrain;

  /** All test instances */
  protected Instances m_Test;

  /** Training pairs */
  protected ArrayList m_labeledTrainPairs; 

  /** The weight of all incorrectly categorized test instances. */
  protected double m_WeightTestIncorrect;

  /** The weight of all correctly categorized test instances. */
  protected double m_WeightTestCorrect;

  /** The weight of all uncategorized test instances. */
  protected double m_WeightTestUnclassified;

  /** The weight of test instances that had a class assigned to them. */
  protected double m_WeightTestWithClass;

  /** Array for storing the confusion matrix. */
  protected double [][] m_ConfusionMatrix;

  /** The names of the classes. */
  protected String [] m_ClassNames;

  /** Is the class nominal or numeric? */
  protected boolean m_ClassIsNominal;

  /** If the class is not nominal, we do not need the confusion matrix but do pairs counts directly */
  protected int m_totalPairs;
  protected int m_goodPairs;
  protected int m_trueGoodPairs;
  
  /** The total cost of predictions (includes instance weights) */
  protected double m_TotalCost;


  public String toSummaryString() {
    return super.toString();
  }

  /**
   * Returns a string describing this evaluator
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return " A clusterer evaluator that evaluates results of running a "
      + "semi-supervised clustering algorithm.";
  }

  public SemiSupClustererEvaluation (Instances test, int numClasses, int numClusters) {
    m_NumClasses = numClasses;
    m_NumClusters = numClusters;
    m_ClassIsNominal = test.classAttribute().isNominal();

    if (m_ClassIsNominal) {
      m_ConfusionMatrix = new double [m_NumClusters][m_NumClasses];
      m_ClassNames = new String [m_NumClasses];
      for(int i = 0; i < m_NumClasses; i++) {
	m_ClassNames[i] = test.classAttribute().value(i);
      }
    } 
  }


  public SemiSupClustererEvaluation (ArrayList labeledTrainPairs, Instances test, int numClasses, int numClusters) {
    this (test,numClasses,numClusters);
    m_labeledTrainPairs = labeledTrainPairs;
  }

  
  /**
   * Evaluates the semi-sup clusterer on a given set of test instances
   *
   * @param clusterer semi-supervised clusterer 
   * @param testInstances set of test instances for evaluation
   * @exception Exception if model could not be evaluated successfully
   */
  public void evaluateModel (Clusterer clusterer, Instances testInstances, Instances unlabeledTest) throws Exception {
    if (m_ClassIsNominal) { 
      m_Test = testInstances;
      m_Objective = ((SemiSupClusterer) clusterer).objectiveFunction(); // Assuming transductive clustering here ... will need to generalize in future
      System.out.println("Evaluating cluster results ...");
      for (int i = 0; i < unlabeledTest.numInstances(); i++) {
	evaluateModelOnce(clusterer, unlabeledTest.instance(i), (int) (testInstances.instance(i)).classValue());
      }
    } else { // string-based class attributes
      int numInstances = testInstances.numInstances();
      Attribute classAttr = testInstances.classAttribute();
      int [][] sharedClass = new int[numInstances][numInstances];
      HashSet dontCareSet = new HashSet();
      final int HAVE_SHARED_CLASS = 0;
      final int NO_SHARED_CLASS = 1;
      final int DONT_CARE = 2; 
      m_totalPairs = 0;
      m_goodPairs = 0;

      // calculate the number of true pairs
      m_trueGoodPairs = 0;
      HashSet [] classSets = new HashSet[numInstances];
      for (int i = 0; i < numInstances; i++) {
	  System.out.println("Classattr: " + classAttr);
	String classList = testInstances.instance(i).stringValue(classAttr);
	if (classList.length() != 0) { // skip unassigned instances
	  // parse the list of classes into a hashset
	  HashSet classSet = new HashSet();
	  StringTokenizer tokenizer = new StringTokenizer(classList, "_");
	  while (tokenizer.hasMoreTokens()) {
	    classSet.add(tokenizer.nextToken());
	  }
	  classSets[i] = classSet;
	  for (int j = 0; j < i; j++) {
	    if (classSets[j] != null) { // skip unassigned instances
	      HashSet prevSet = (HashSet) classSets[j];
	      Iterator iterator = prevSet.iterator();
	      boolean shareClass = false;

	      // go through previously assigned instance's classes and see if current class list contains any
	      while (iterator.hasNext() && !shareClass) {
		String classString = (String) iterator.next();
		if (classSet.contains(classString)) {
		  shareClass = true;
		} 
	      }
	      if (shareClass) {
		m_trueGoodPairs++;
		sharedClass[i][j] = sharedClass[j][i] = HAVE_SHARED_CLASS;
	      } else {
		sharedClass[i][j] = sharedClass[j][i] = NO_SHARED_CLASS;
	      } 
	    }
	  }
	} else { // all pairs with this instance are don't care
	  dontCareSet.add(new Integer(i));
	  for (int j = 0; j < numInstances; j++) {
	    sharedClass[i][j] = sharedClass[j][i] = DONT_CARE;
	  } 
	} 
      }

      // now cluster and evaluate precision
      ArrayList[] classLists = new ArrayList[m_NumClasses];
      for (int i = 0; i < classLists.length; i++) {
	classLists[i] = new ArrayList();
      } 
      for (int i = 0; i < unlabeledTest.numInstances(); i++) {
	if (!dontCareSet.contains(new Integer(i))) { 
	  int clusterIdx = clusterer.clusterInstance(unlabeledTest.instance(i));

	  // go through all instances previously assigned to the same cluster and check whether they have common classes
	  for (int j = 0; j < classLists[clusterIdx].size(); j++) {
	    int sameClusterInstanceIdx = ((Integer) classLists[clusterIdx].get(j)).intValue();
	    if (sharedClass[j][sameClusterInstanceIdx] == HAVE_SHARED_CLASS) {
	      m_goodPairs++;
	    } 
	    m_totalPairs++; 
	  }
	  classLists[clusterIdx].add(new Integer(i));
	}
      }
    } 
  }

  /**
   * Evaluates the semi-sup clusterer on a given test instance
   *
   * @param clusterer semi-supervised clusterer 
   * @param test test instance for evaluation
   * @exception Exception if model could not be evaluated successfully
   */
  public void evaluateModelOnce (Clusterer clusterer, Instance testWithoutLabel, int classValue) throws Exception {
    double [] pred;
    if (m_ClassIsNominal) {
      if (clusterer instanceof DistributionClusterer) {
	pred = ((DistributionClusterer) clusterer).distributionForInstance(testWithoutLabel);
      }
      else {
	pred = makeDistribution(clusterer.clusterInstance(testWithoutLabel));
      }
      updateStatsForClusterer(pred, classValue);
    }
  }

  /**
   * Convert a single prediction into a probability distribution
   * with all zero probabilities except the predicted value which
   * has probability 1.0;
   *
   * @param predictedClass the index of the predicted class
   * @return the probability distribution
   */
  protected double [] makeDistribution(int predictedCluster) {

    double [] result = new double [m_NumClasses];

    if (m_ClassIsNominal) {
      result[predictedCluster] = 1.0;
    } 
    else {
      result[0] = predictedCluster;
    }
    return result;
  } 

  /**
   * Updates all the statistics about a clusterer performance for 
   * the current test instance.
   *
   * @param distrib the probabilities assigned to each class
   * @param test the test instance
   * @exception Exception if the class of the instance is not set
   */
  protected void updateStatsForClusterer(double [] distrib, int classValue) {
    for (int i=0; i<distrib.length; i++) {
      //      System.out.println("Adding value to distrib: " + i + " with classValue: " + classValue);
      m_ConfusionMatrix[i][classValue] += distrib[i];
    }
  }

  public final double objectiveFunction() {
    return m_Objective;
  }

  public final double purity() {
    return m_Purity;
  }

  public final double entropy() {
    return m_Entropy;
  }

  public final double klDivergence() {
    return m_KLDivergence;
  }

  public final double mutualInformation() {
    if (m_ClassIsNominal) { 
      double [] clusterTotals = new double[m_NumClusters];
      double [] classTotals = new double[m_NumClasses];

      for (int i=0; i<m_NumClusters; i++) {
	for (int j=0; j<m_NumClasses; j++) {
	  clusterTotals[i] += m_ConfusionMatrix[i][j];
	  classTotals[j] += m_ConfusionMatrix[i][j];
	}
      }

      try {
	System.out.println(toMatrixString("\nConfusion matrix:")); 
      } catch(Exception e) {
	e.printStackTrace();
      }

      // calculate MI from counts
      m_MIMetric = 0.0;
      int numInstances = m_Test.numInstances();

      double MI = 0;
      for (int i=0; i<m_NumClusters; i++) {
	for (int j=0; j<m_NumClasses; j++) {
	  if(m_ConfusionMatrix[i][j] !=0 && clusterTotals[i] != 0 && classTotals[i] != 0) {
	    if (clusterTotals[i] != 0 && classTotals[j] != 0) { 
	      MI += (1.0 * m_ConfusionMatrix[i][j]/numInstances)
		* Math.log((1.0 * m_ConfusionMatrix[i][j] * numInstances) 
			   / (clusterTotals[i] * classTotals[j]));
	    } 
	  }			    
	}
      }
      double classEntropy = 0, clusterEntropy = 0;
      for (int i=0; i<m_NumClusters; i++) {
	if (clusterTotals[i] != 0) { 
	  clusterEntropy -= (1.0 * clusterTotals[i])/numInstances 
	    * Math.log(1.0 * clusterTotals[i]/numInstances);
	}
      }	
      for (int j=0; j<m_NumClasses; j++) {
	if (classTotals[j] != 0) { 
	  classEntropy -= (1.0 * classTotals[j])/numInstances 
	    * Math.log(1.0 * classTotals[j]/numInstances);
	}
      }

      m_MIMetric = 2*MI / (classEntropy + clusterEntropy);
      System.out.println("Final MI is: " + m_MIMetric + "\t" + classEntropy + "\t" + clusterEntropy);
    }
    return m_MIMetric;
  }


    /**
   * Outputs the performance statistics as a classification confusion
   * matrix. For each class value, shows the distribution of 
   * predicted class values.
   *
   * @param title the title for the confusion matrix
   * @return the confusion matrix as a String
   * @exception Exception if the class is numeric
   */
  public String toMatrixString(String title) throws Exception {

    StringBuffer text = new StringBuffer();
    char [] IDChars = {'a','b','c','d','e','f','g','h','i','j',
		       'k','l','m','n','o','p','q','r','s','t',
		       'u','v','w','x','y','z'};
    int IDWidth;
    boolean fractional = false;

    // Find the maximum value in the matrix
    // and check for fractional display requirement 
    double maxval = 0;
    for(int i = 0; i < m_NumClusters; i++) {
      for(int j = 0; j < m_NumClasses; j++) {
	double current = m_ConfusionMatrix[i][j];
        if (current < 0) {
          current *= -10;
        }
	if (current > maxval) {
	  maxval = current;
	}
	double fract = current - Math.rint(current);
	if (!fractional
	    && ((Math.log(fract) / Math.log(10)) >= -2)) {
	  fractional = true;
	}
      }
    }

    IDWidth = 1 + Math.max((int)(Math.log(maxval) / Math.log(10) 
				 + (fractional ? 3 : 0)),
			     (int)(Math.log(m_NumClasses) / 
				   Math.log(IDChars.length)));
    text.append(title).append("\n");
    for(int i = 0; i < m_NumClasses; i++) {
      if (fractional) {
	text.append(" ").append(num2ShortID(i,IDChars,IDWidth - 3))
          .append("   ");
      } else {
	text.append(" ").append(num2ShortID(i,IDChars,IDWidth));
      }
    }
    text.append("   <-- classes; rows=clusters\n");
    for(int i = 0; i< m_NumClusters; i++) { 
      for(int j = 0; j < m_NumClasses; j++) {
	text.append(" ").append(
		    Utils.doubleToString(m_ConfusionMatrix[i][j],
					 IDWidth,
					 (fractional ? 2 : 0)));
      }
      text.append(" | ").append(num2ShortID(i,IDChars,IDWidth))
        .append(" = ").append(m_ClassNames[i]).append("\n");
    }
    return text.toString();
  }

    /**
   * Method for generating indices for the confusion matrix.
   *
   * @param num integer to format
   * @return the formatted integer as a string
   */
  private String num2ShortID(int num,char [] IDChars,int IDWidth) {
    
    char ID [] = new char [IDWidth];
    int i;
    
    for(i = IDWidth - 1; i >=0; i--) {
      ID[i] = IDChars[num % IDChars.length];
      num = num / IDChars.length - 1;
      if (num < 0) {
	break;
      }
    }
    for(i--; i >= 0; i--) {
      ID[i] = ' ';
    }

    return new String(ID);
  }

  
  public final double pairwisePrecision() {
    if (m_ClassIsNominal) { 
      int [] clusterTotals = new int[m_NumClusters];
      int [] goodPairTotals = new int[m_NumClusters];
      m_totalPairs = 0;
      m_goodPairs = 0;
    
      for (int i = 0; i < m_NumClusters; i++) {
	for (int j = 0; j < m_NumClasses; j++) {
	  goodPairTotals[i] += m_ConfusionMatrix[i][j] * (m_ConfusionMatrix[i][j] - 1) / 2;
	  clusterTotals[i] += m_ConfusionMatrix[i][j];
	}
      }

      for (int i = 0; i < m_NumClusters; i++) {
	m_totalPairs += clusterTotals[i] * (clusterTotals[i] - 1) / 2;
	m_goodPairs += goodPairTotals[i];
      }
    }
    return (m_goodPairs+0.0)/m_totalPairs;
  }

  public final double pairwiseRecall() {
    if (m_ClassIsNominal) { 
      int [] classTotals = new int[m_NumClasses];
      int [] goodPairTotals = new int[m_NumClasses];
      m_trueGoodPairs = 0;
      m_goodPairs = 0;
    
      for (int i = 0; i < m_NumClasses; i++) {
	for (int j = 0; j < m_NumClusters; j++) {
	  goodPairTotals[i] += m_ConfusionMatrix[j][i] * (m_ConfusionMatrix[j][i] - 1) / 2;
	  classTotals[i] += m_ConfusionMatrix[j][i];
	}
      }

      for (int i = 0; i < m_NumClasses; i++) {
	m_trueGoodPairs += classTotals[i] * (classTotals[i] - 1) / 2;
	m_goodPairs += goodPairTotals[i];
      }
    }
    return (m_goodPairs+0.0)/m_trueGoodPairs;
  }

  public final double pairwiseFMeasure() {
    double fmeasure = 0;
    if (m_ClassIsNominal) { 
      int [] clusterTotals = new int[m_NumClusters];
      int [] classTotals = new int[m_NumClasses];
      int [] goodPairTotals = new int[m_NumClusters];
      int totalClassPairs = 0;
      int totalClusterPairs = 0;
      int goodPairs = 0;
    
      for (int i = 0; i < m_NumClusters; i++) {
	for (int j = 0; j < m_NumClasses; j++) {
	  goodPairTotals[i] += m_ConfusionMatrix[i][j] * (m_ConfusionMatrix[i][j] - 1) / 2;
	  clusterTotals[i] += m_ConfusionMatrix[i][j];
	  classTotals[j] += m_ConfusionMatrix[i][j];
	}
      }

      for (int i = 0; i < m_NumClusters; i++) {
	totalClusterPairs += clusterTotals[i] * (clusterTotals[i] - 1) / 2;
	goodPairs += goodPairTotals[i];
      }
      for (int i = 0; i < m_NumClasses; i++) {
	totalClassPairs += classTotals[i] * (classTotals[i] - 1) / 2;
      }
      double precision = (goodPairs+0.0)/totalClusterPairs;
      double recall = (goodPairs+0.0)/totalClassPairs;

      if (precision > 0) {  // avoid divide by zero in the p=0&r=0 case
	fmeasure = 2 * (precision * recall) / (precision + recall);
      }
      System.out.println("Final F-Measure is: " + fmeasure + "; Precision=" + precision + "  Recall=" + recall + "\n");
    } else { // the class is not nominal
      fmeasure = 2.0 * m_goodPairs / (m_totalPairs + m_trueGoodPairs);
    } 
    return fmeasure;
  }

  public final double numSameClassPairs() {
    int numSameClassPairs = 0;
    for (int i = 0; i < m_labeledTrainPairs.size(); i++) {
      InstancePair pair = (InstancePair) m_labeledTrainPairs.get(i);
      if (pair.linkType == InstancePair.MUST_LINK) {
	numSameClassPairs++;
      }
    }
    return numSameClassPairs;
  }

  public final double numDiffClassPairs() {
    int numDiffClassPairs = 0;
    for (int i = 0; i < m_labeledTrainPairs.size(); i++) {
      InstancePair pair = (InstancePair) m_labeledTrainPairs.get(i);
      if (pair.linkType == InstancePair.CANNOT_LINK) {
	numDiffClassPairs++;
      }
    }
    return numDiffClassPairs;
  }

}



