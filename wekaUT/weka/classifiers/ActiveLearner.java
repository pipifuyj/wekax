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
 *    ActiveLearner.java
 *    Copyright (C) 2003 Prem Melville
 *
 */

package weka.classifiers;

import weka.core.*;

/**
 * Interface to permit a classifier to perform selective sampling. A
 * classifier that implements this interface can actively select
 * training examples from a given set of unlabeled examples.
 *
 * @author Prem Melville (melville@cs.utexas.edu)
 * @version 
 */
public interface ActiveLearner {
    
    /** 
     * Given a set of unlabeled examples, select a specified number of examples to be labeled.
     * @param unlabeledActivePool pool of unlabeled examples
     * @param num number of examples to selcted for labeling
     * @exception Exception if selective sampling fails
     */
    int [] selectInstances(Instances unlabeledActivePool,int num) throws Exception;



}
 
