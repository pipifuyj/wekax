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
 *    TestEnsembleClassifier
 *    Copyright (C) 2003 Prem Melville
 *
 */

package weka.classifiers.meta;

import weka.classifiers.*;
import java.util.*;
import weka.core.*;

/**
 * This class is for testing Ensemble evaluation
 */
public class TestEnsembleClassifier extends EnsembleClassifier{
 
    protected int m_NumIterations=21;
    protected Random random = new Random();
    
  /**
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @exception Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {
      //Initialize measures
      initMeasures();
      
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
      double [] sums = new double [instance.numClasses()];
      double [] preds = getEnsemblePredictions(instance);
    
      for (int i = 0; i < m_NumIterations; i++) {
	  sums[(int)preds[i]]++;
      }
      
      Utils.normalize(sums);
      return sums;
  }
    
    /** Returns class predictions of each ensemble member */
    public double []getEnsemblePredictions(Instance instance) throws Exception{
	double preds[] = new double [m_NumIterations];
	double actualClass;
	
	if(instance.classIsMissing()) {
	    actualClass = random.nextInt(instance.numClasses());
	    //for(int i=0; i<m_NumIterations; i++) preds[i] = actualClass;
	    for(int i=0; i<m_NumIterations; i++) preds[i] = 1.0;
	}
	else {
	    actualClass = instance.classValue();
	    
	    for(int i=0; i<m_NumIterations; i++){
		if(random.nextFloat()<0.4)
		    preds[i] = actualClass;
		else
		    preds[i] = (actualClass+1)%instance.numClasses();
	    }
	}
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

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
   
    try {
      System.out.println(Evaluation.
			 evaluateModel(new Bagging(), argv));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
}




