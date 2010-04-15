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
 *    ExtractionEvaluation.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package  weka.extraction;

import  java.util.*;
import  java.io.*;
import  weka.core.*;
import  weka.filters.Filter;
import  weka.filters.unsupervised.attribute.Remove;

/**
 * Class for evaluating extractors
 *
 * @author  Mikhail Bilenko (mbilenko@cs.utexas.edu)
 */
public class ExtractionEvaluation {

  /** Training instances */
  protected Instances m_trainInstances;
  
  /** Test instances */
  protected Instances m_testInstances;

  /**
   * Returns a string describing this evaluator
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return " A extraction evaluator that evaluates results of running a "
      + "extraction experiment.";
  }

  /** A default constructor */
  public ExtractionEvaluation () {
  }

  /** Train an extractor on supplied data
   * @param extractor the extractor to train
   * @param labeledData data that is labeled for training the extractor
   * @param unlabeledData unlabeled data for transductive extractors
   */
  public void trainExtractor(Extractor extractor, Instances labeledData, Instances unlabeledData) throws Exception {
    extractor.trainExtractor(labeledData, unlabeledData);
  } 

  /**
   * Evaluates an extractor on a given set of test instances
   *
   * @param extractor the extractor to evaluate
   * @param testData set of test instances for evaluation
   * @return a list of arrays containing the basic statistics for each point
   * @exception Exception if model could not be evaluated successfully
   */
  public ArrayList evaluateModel (Extractor extractor, Instances testData) throws Exception {
    // Run the extractor collecting data
    HashMap docFillerMap = createDocFillerMap(testData);
    extractor.testExtractor(testData, docFillerMap);
    return extractor.getStatistics();
  }


  /**
   * Given a set of data, create a HashMap which maps each Instance's uniqueID
   * to a fillerPositionListMap.  In that map, every filler is mapped to a list of
   * positions where it should extracted.
   */
  protected HashMap createDocFillerMap(Instances data) {
    HashMap docFillerMap = new HashMap();

    Attribute uniqueIDAttr = data.attribute("uniqueID");
    Attribute textAttr = data.attribute("text");
    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i);
      String uniqueID = instance.stringValue(uniqueIDAttr);
      String text = instance.stringValue(textAttr);

      HashMap fillerPositionListMap = new HashMap();
      // TODO:  go through text, and create a map where each
      // filler is mapped to a list of positions where it occurs

      docFillerMap.put(uniqueID, fillerPositionListMap);
    } 
    return docFillerMap;
  } 
}
