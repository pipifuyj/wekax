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
 *    MPCKMeans.java
 *    Copyright (C) 2003 Sugato Basu and Misha Bilenko
 *
 */
package weka.clusterers;

import  java.io.*;
import  java.util.*;
import  weka.core.*;

/**
 * Utils useful for clustering
 */
public class ClusterUtils {

  /** Normalizes Instance or SparseInstance
   *
   * @author Sugato Basu
   * @param inst Instance to be normalized
   */

  public static void normalize(Instance inst) throws Exception {
    if (inst instanceof SparseInstance) {
      normalizeSparseInstance(inst);
    }
    else {
      normalizeInstance(inst);
    }
  }

  /** Normalizes the values of a normal Instance in L2 norm
   *
   * @author Sugato Basu
   * @param inst Instance to be normalized
   */

  public static void normalizeInstance(Instance inst) throws Exception{
    double norm = 0;
    double values [] = inst.toDoubleArray();

    if (inst instanceof SparseInstance) {
      System.err.println("Is SparseInstance, using normalizeSparseInstance function instead");
      normalizeSparseInstance(inst);
    }
    
    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	norm += values[i] * values[i];
      }
    }
    norm = Math.sqrt(norm);
    for (int i=0; i<values.length; i++) {
      if (i != inst.classIndex()) { // don't normalize the class index 
	values[i] /= norm;
      }
    }
    inst.setValueArray(values);
  }

  /** Normalizes the values of a SparseInstance in L2 norm
   *
   * @author Sugato Basu
   * @param inst SparseInstance to be normalized
   */

  public static void normalizeSparseInstance(Instance inst) throws Exception{
    double norm=0;
    int length = inst.numValues();

    if (!(inst instanceof SparseInstance)) {
      System.err.println("Not SparseInstance, using normalizeInstance function instead");
      normalizeInstance(inst);
    }

    for (int i=0; i<length; i++) {
      if (inst.index(i) != inst.classIndex()) { // don't normalize the class index
	norm += inst.valueSparse(i) * inst.valueSparse(i);
      }
    }
    norm = Math.sqrt(norm);
    for (int i=0; i<length; i++) { // don't normalize the class index
      if (inst.index(i) != inst.classIndex()) {
	inst.setValueSparse(i, inst.valueSparse(i)/norm);
      }
    }
  }

  /** Normalize an array of double's
   */
  public static double[] normalize(double[] weights) {
    double sum = 0;
    for (int i = 0; i < weights.length; i++) {
      sum += weights[i];
    }
    if (sum != 0) { 
      for(int i = 0; i < weights.length; i++) {
	weights[i] = weights[i] / sum; 
      }
    }
    return weights; 
  } 
  
  /** Fast version of meanOrMode - streamlined from Instances.meanOrMode for efficiency 
   *  Does not check for missing attributes, assumes numeric attributes, assumes Sparse instances
   */

  public static double[] meanOrMode(Instances insts) {

    int numAttributes = insts.numAttributes();
    double [] value = new double[numAttributes];
    double weight = 0;
    
    for (int i=0; i<numAttributes; i++) {
      value[i] = 0;
    }

    for (int j=0; j<insts.numInstances(); j++) {
      SparseInstance inst = (SparseInstance) (insts.instance(j));
      weight += inst.weight();

      for (int i=0; i<inst.numValues(); i++) {
	int indexOfIndex = inst.index(i);
	value[indexOfIndex]  += inst.weight() * inst.valueSparse(i);
      }
    }
    
    if (Utils.eq(weight, 0)) {
      for (int k=0; k<numAttributes; k++) {
	value[k] = 0;
      }
    }
    else {
      for (int k=0; k<numAttributes; k++) {
	value[k] = value[k] / weight;
      }
    }

    return value;
  }

  /** This function divides every attribute value in an instance by
   *  the instance weight -- useful to find the mean of a cluster in
   *  Euclidean space 
   *  @param inst Instance passed in for normalization (destructive update)
   */
  public static void normalizeByWeight(Instance inst) {
    double weight = inst.weight();
    if (inst instanceof SparseInstance) { 
      for (int i=0; i<inst.numValues(); i++) {
	inst.setValueSparse(i, inst.valueSparse(i)/weight);
      }
    }
    else if (!(inst instanceof SparseInstance)) {
      for (int i=0; i<inst.numAttributes(); i++) {
	inst.setValue(i, inst.value(i)/weight);
      }
    }
  }


  /** Finds the sum of instance sum with instance inst 
   */
  public static Instance sumWithInstance(Instance sum, Instance inst, Instances m_Instances) throws Exception {
    Instance newSum;
    if (sum == null) {
      if (inst instanceof SparseInstance) {
	newSum = new SparseInstance(inst);
	newSum.setDataset(m_Instances);
      }
      else {
	newSum = new Instance(inst);
	newSum.setDataset(m_Instances);
      }
    }
    else {
      newSum = sumInstances(sum, inst, m_Instances);
    }
    return newSum;
  }


  /** Finds sum of 2 instances (handles sparse and non-sparse)
   */

  public static Instance sumInstances(Instance inst1, Instance inst2, Instances m_Instances) throws Exception {
    int numAttributes = inst1.numAttributes();
    if (inst2.numAttributes() != numAttributes) {
      throw new Exception ("Error!! inst1 and inst2 should have same number of attributes.");
    }
    double weight1 = inst1.weight(), weight2 = inst2.weight();
    double [] values = new double[numAttributes];
    
    for (int i=0; i<numAttributes; i++) {
      values[i] = 0;
    }
    
    if (inst1 instanceof SparseInstance && inst2 instanceof SparseInstance) {
      for (int i=0; i<inst1.numValues(); i++) {
	int indexOfIndex = inst1.index(i);
	values[indexOfIndex] = inst1.valueSparse(i);
      }
      for (int i=0; i<inst2.numValues(); i++) {
	int indexOfIndex = inst2.index(i);
	values[indexOfIndex] += inst2.valueSparse(i);
      }
      SparseInstance newInst = new SparseInstance(weight1+weight2, values);
      newInst.setDataset(m_Instances);
      return newInst;
    }
    else if (!(inst1 instanceof SparseInstance) && !(inst2 instanceof SparseInstance)){
      for (int i=0; i<numAttributes; i++) {
	values[i] = inst1.value(i) + inst2.value(i);
      }
    }
    else {
      throw new Exception ("Error!! inst1 and inst2 should be both of same type -- sparse or non-sparse");
    }
    Instance newInst = new Instance(weight1+weight2, values);
    newInst.setDataset(m_Instances);
    return newInst;
  }


  /**
   * Gets a Double representing the current date and time.
   * eg: 1:46pm on 20/5/1999 -> 19990520.1346
   *
   * @return a value of type Double
   */
  public static Double getTimeStamp() {

    Calendar now = Calendar.getInstance(TimeZone.getTimeZone("UTC"));
    double timestamp = now.getTimeInMillis();
    return new Double(timestamp);
  }
}
