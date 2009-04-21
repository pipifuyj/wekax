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
 *    EnsembleClassifier.java
 *    Copyright (C) 2003 Prem Melville
 *
 */

package weka.classifiers;

import weka.core.*;
import com.jmage.*;
import java.util.*;

/** 
 * Abstract class for Ensemble Classifiers
 *
 * @author Prem Melville
 * @version $Revision: 1.3 $
 */
public abstract class EnsembleClassifier extends DistributionClassifier implements AdditionalMeasureProducer{
    
    /** the error on the training data */
    protected double m_TrainError=0;
    /** the average error of the ensemble on the training data */
    protected double m_TrainEnsembleError=0;
    /** the ensemble diversity computed in the training data */
    protected double m_TrainEnsembleDiversity=0;
    /** Sum of ensemble weights */
    protected double m_SumEnsembleWts=0;
    /** Vote weights of ensemble members */
    protected double []m_EnsembleWts;

    /** Returns class predictions of each ensemble member */
    public abstract double []getEnsemblePredictions(Instance instance) 
	throws Exception;
    
    /** 
     * Returns vote weights of ensemble members.
     *
     * @return vote weights of ensemble members
     */
    public abstract double []getEnsembleWts();
    
    /** Returns size of ensemble */
    public abstract double getEnsembleSize();
  /**
   * Returns an enumeration of the additional measure names
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    Vector newVector = new Vector(3);
    newVector.addElement("measureTrainError");
    newVector.addElement("measureTrainEnsembleError");
    newVector.addElement("measureTrainEnsembleDiversity");
    return newVector.elements();
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @exception IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    if (additionalMeasureName.compareTo("measureTrainError") == 0) {
	return measureTrainError();
    } else if (additionalMeasureName.compareTo("measureTrainEnsembleError") == 0) {
	return measureTrainEnsembleError();
    } else if (additionalMeasureName.compareTo("measureTrainEnsembleDiversity") == 0) {
	return measureTrainEnsembleDiversity();
    } else {
      throw new IllegalArgumentException(additionalMeasureName 
			  + " not supported (DEC)");
    }
  }
    
    /**
     * @return the error on the training data
     **/
    public double measureTrainError(){
	return m_TrainError;
    }
    
    /** 
     * @return the average error of the ensemble on the training data
     */
    public double measureTrainEnsembleError(){
	return m_TrainEnsembleError;
    }
    
    /**
     * @return the ensemble diversity computed in the training data
     */
    public double measureTrainEnsembleDiversity(){
	return m_TrainEnsembleDiversity;
    }
    

    /** Initialize measures */
    protected void initMeasures(){
	m_SumEnsembleWts=0;
	m_TrainError=0;
	m_TrainEnsembleError=0;
	m_TrainEnsembleDiversity=0;
    }
    

    /**
     * Compute ensemble measures.
     * @param data training instances
     */
    protected void computeEnsembleMeasures(Instances data) throws Exception{
	for(int j=0; j<getEnsembleSize(); j++)
	    m_SumEnsembleWts += m_EnsembleWts[j];
	
	//DEBUG
	//System.out.println("Ensemble size = "+getEnsembleSize());
	if(m_SumEnsembleWts == 0.0){
	    System.out.println("Ensemble wts sum to 0!");
	    for(int j=0; j<m_EnsembleWts.length; j++)
		System.out.print("\t"+m_EnsembleWts[j]);
	    System.out.println();
	}
	
	double totalInstanceWt=0;
	Instance curr;
	for (int i = 0; i < data.numInstances(); i++) {
	    curr = data.instance(i); 
	    totalInstanceWt += curr.weight();
	    if(curr.weight() != 1.0) System.out.println(">>> Instance Weight = "+curr.weight());
	    updateEnsembleStats(classifyInstance(curr), curr, getEnsemblePredictions(curr));
	}
	//DEBUG
	Assert.that(totalInstanceWt==data.numInstances(),"Instance wts don't total to num of instances!");
	
	m_TrainError = 100.0 * (m_TrainError/totalInstanceWt);
	m_TrainEnsembleError = 100.0 * m_TrainEnsembleError/totalInstanceWt;
	m_TrainEnsembleDiversity = 100.0 * m_TrainEnsembleDiversity/totalInstanceWt;
    }
    
    /**
     * Update statistics for ensemble classifiers.
     *
     * @param pred ensemble prediction
     * @param instance training instance
     * @param ensemblePreds predictions of ensemble members
     */
    protected void updateEnsembleStats(double pred, Instance instance, double []ensemblePreds){
	//System.out.print("Updating Ensemble Stats...");

	double sumEnsembleError = 0, sumEnsembleDiversity = 0;
	double actualClass = instance.classValue();
	
	for(int i=0; i<getEnsembleSize(); i++){
	    if(actualClass != ensemblePreds[i])
		sumEnsembleError += m_EnsembleWts[i];
	    
	    //if member's prediction differs from the ensemble prediction, diversity increases 
	    if(pred != ensemblePreds[i])
		sumEnsembleDiversity += m_EnsembleWts[i];
	}
	
	if(pred != actualClass) m_TrainError += instance.weight();
	m_TrainEnsembleError += ((sumEnsembleError/m_SumEnsembleWts)*instance.weight());
	m_TrainEnsembleDiversity += ((sumEnsembleDiversity/m_SumEnsembleWts)*instance.weight());
    }
}
