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
 *    BudgetedLearner.java
 *    Copyright (C) 2005 Prem Melville
 *
 */

////////////////////////////////
//
// WARNING: UNDER DEVELOPMENT
//
////////////////////////////////

package weka.classifiers;

import weka.core.*;

/**
 * Interface to permit a classifier to make instance-feature queries.
 *
 * @author Prem Melville (melville@cs.utexas.edu)
 * @version 
 */
public interface BudgetedLearner {
    
    //Set costs of acquiring each feature 
    void setFeatureCosts(double []featureCosts);
    
    /** 
     * Given a set of incomplete instances, select a specified number of instance-feature queries.
     * @param train set of incomplete instances
     * @param num number of instance-feature pairs to selcted for acquiring remaining features
     * @param queryMatrix matrix to track available queries
     * @exception Exception if selection fails
     */
    Pair []selectInstancesForFeatures(Instances train, int num, boolean [][]queryMatrix) throws Exception;

}
 

