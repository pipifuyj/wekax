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
 *    Seeder.java
 *    Copyright (C) 2001 Sugato Basu, Mikhail Bilenko
 *
 */

package weka.clusterers;

import weka.core.*;
import java.util.*;

public class Seeder extends ArrayList {

  /** Stores the mapping between all possible seeds and their cluster assignments */
  protected HashMap m_TotalSeedHash;
  
  /** Stores the current instances which are set as seeds */
  protected ArrayList m_CurrentSeedInstances;

  /** Verbose? */
  protected boolean m_Verbose = false;
  
  /* Constructor */
  public Seeder(HashMap totalSeedHash) {
    m_TotalSeedHash = totalSeedHash;
  }

  /**
   * set the verbosity level of the clusterer
   * @param verbose messages on(true) or off (false)
   */
  public void setVerbose (boolean verbose) {
    m_Verbose = verbose;
  }

  /** Constructor 
   *  @param dataWithClass: Data set which has class information in it
   *  @param dataWithoutClass: dataWithClass dataset with class information removed 
   */
  public Seeder(Instances dataWithoutClass, Instances dataWithClass) throws Exception{
    int hashSize = (int) (dataWithClass.numInstances()/0.75 + 10); // Java API recommendations    
    int classIndex = dataWithClass.classIndex();

    m_TotalSeedHash = new HashMap (hashSize);
    if (classIndex < 0) {
      throw new WekaException ("Need class information in data set");
    }
    if (dataWithClass.numInstances() != dataWithoutClass.numInstances()) {
      throw new WekaException ("Both datasets should have same size");
    }
      
    if (m_Verbose) {
      System.out.println("Total seed hash table ...\n");
    }
  
    for (int i = 0; i < dataWithoutClass.numInstances(); i++) {
      Instance instWithClass = dataWithClass.instance(i);
      Instance instWithoutClass = dataWithoutClass.instance(i);
      m_TotalSeedHash.put(instWithoutClass, new Integer((int) instWithClass.classValue()));
      if (m_Verbose) {
	System.out.println("Inserting key: " + instWithoutClass + " and value: " + instWithClass.classValue());
      }
    }
  }

  /** Set the current seeds */
  public void createSeeds (ArrayList seed_data) {
    m_CurrentSeedInstances = seed_data;
  }


  /** Returns the total hashMap, with the instance to cluster assignment mapping for all the seeds 
   *
   * @return the total hashMap 
   */
  
  public HashMap getAllSeeds() throws Exception {
    return m_TotalSeedHash;
  }

  /** Returns a hashMap with the instance to cluster assignment mapping for the current seeds 
   *
   * @return the seed hashMap 
   */
  
  public HashMap getSeeds() throws Exception {
    int hashSize = (int) (m_CurrentSeedInstances.size()/0.75 + 10); // Java API recommendations
    HashMap returnHash = new HashMap(hashSize);

    for (int i=0; i<m_CurrentSeedInstances.size(); i++) {
      Instance seed = (Instance) m_CurrentSeedInstances.get(i);
      if(!m_TotalSeedHash.containsKey(seed))
	throw new Exception("Seed does not have an entry in the totalSeedHash");
      returnHash.put(seed, m_TotalSeedHash.get(seed));
    }
    return returnHash;
  }
}
