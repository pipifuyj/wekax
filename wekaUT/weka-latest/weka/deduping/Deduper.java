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
 *    Deduper.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.deduping;

import weka.core.Instances;
import weka.core.Utils;
import java.util.ArrayList;

/** An abstract class that takes a set of objects and
 * identifies disjoint subsets of duplicates
 *
 * @author Mikhail Bilenko (mbilenko@cs.utexas.edu)
 * @version $Revision: 1.1 $
 */
public abstract class Deduper implements Cloneable {

  /** An arraylist of Object arrays containing statistics */
  protected ArrayList m_statistics = null;

  /** Given training data, build the metrics required by the deduper
   * @param train a set of training data
   */
  public abstract void buildDeduper(Instances trainInstances, Instances testInstances) throws Exception;


  /** Identify duplicates within the testing data
   * @param testInstances a set of instances among which to identify duplicates
   * @param numObjects the number of "true object" sets to create
   */
  public abstract void findDuplicates(Instances testInstances, int numObjects) throws Exception;

  public static Deduper forName(String deduperName, String [] options) throws Exception {
    return (Deduper)Utils.forName(Deduper.class,
				  deduperName,
				  options);
  }

    /** Return the list of statistics collected during deduping
   * @returns collected statistics
   */
  public ArrayList getStatistics() {
    return m_statistics;
  }

}



