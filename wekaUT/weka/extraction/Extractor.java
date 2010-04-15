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
 *    Extractor.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.extraction;

import weka.core.Instances;
import weka.core.Utils;
import java.util.HashMap;
import java.util.ArrayList;

/** An abstract extractor class. Takes a set of objects and trains on it;
 * then can be used for extraction on a testing set.
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */
public abstract class Extractor implements Cloneable {

  /** An arraylist of Object arrays containing statistics */
  protected ArrayList m_statistics = null;

  /** Given training data, train the extractor
   * @param labeledData a set of training data
   * @param unlabeledData a set of unlabeled data; used only by extractors
   * that implement transductive learning
   */
  public abstract void trainExtractor(Instances labeledData, Instances unlabeledData) throws Exception;


  /** Perform extraction on a set of data. 
   * @param testData a set of instances on which to perform extraction
   * @param docFillerMap a map where the uniqueID of an instance (document) is mapped to a
   * HashMap, which maps fillers to a list of Integer positions
   */
  public abstract void testExtractor(Instances testData, HashMap docFillerMap) throws Exception;

  
  /** Return the list of statistics collected during extraction
   * @returns a list of Object[], containing collected statistics
   */
  public ArrayList getStatistics() {
    return m_statistics;
  }


  /** A helper function that may be needed by command-line Weka
   */
  public static Extractor forName(String extractorName, String [] options) throws Exception {
    return (Extractor)Utils.forName(Extractor.class,
				  extractorName,
				  options);
  }

}









