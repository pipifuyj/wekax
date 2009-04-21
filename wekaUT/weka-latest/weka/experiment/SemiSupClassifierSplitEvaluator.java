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
 *    SemiSupClassifierSplitEvaluator.java
 *    Copyright (C) 2003 Prem Melville
 *
 */


package weka.experiment;

import java.io.*;
import java.util.*;

import weka.core.*;
import weka.classifiers.*;


/**
 * A SplitEvaluator that produces results for a semi-supervised
 * classification scheme on a nominal class attribute. Currently this
 * evaluator collects the statistics as for purely supervised
 * classifiers. However, it can be modified to collect more statistics
 * specific to semi-supervised learning.
 *
 * -W classname <br>
 * Specify the full class name of the classifier to evaluate. <p>
 *
 * -C class index <br>
 * The index of the class for which IR statistics are to
 * be output. (default 1) <p>
 *
 * @author Prem Melville (melville@cs.utexas.edu) */
public class SemiSupClassifierSplitEvaluator extends ClassifierSplitEvaluator implements SemiSupSplitEvaluator {
    
    /**
     * Gets the results for the supplied train and test datasets.
     *
     * @param train the training Instances.
     * @param unlabeled the unlabeled training Instances.
     * @param test the testing Instances.
     * @return the results stored in an array. The objects stored in
     * the array may be Strings, Doubles, or null (for the missing value).
     * @exception Exception if a problem occurs while getting the results
     */
    public Object [] getResult(Instances train, Instances unlabeled, Instances test) throws Exception{
	if (m_Classifier == null) {
	    throw new Exception("No classifier has been specified");
	}

	//Modification to allow for semisupervision
	if(m_Classifier instanceof SemiSupClassifier) ((SemiSupClassifier) m_Classifier).setUnlabeled(unlabeled);
	
	return(getResult(train, test));
    }
}



