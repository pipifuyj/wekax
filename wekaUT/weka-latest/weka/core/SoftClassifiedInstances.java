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
 *    SoftClassifiedInstances.java
 *    Copyright (C) 2003 Ray Mooney
 *
 */

package weka.core;

import java.util.*;
import java.io.*;

/**
 * An set of Instances that has a probability distribution across class values.
 * Particularly useful for EM using a SoftClassifier
 *
 * @author Ray Mooney (mooney@cs.utexas.edu)
*/

public class SoftClassifiedInstances extends Instances {
  
    /**
     * Create a set of SoftClassifiedInstances from a given set of
     * Instances but with random class probabilities.
   */
    public SoftClassifiedInstances (Instances dataset, Random randomizer) {
	super(dataset, dataset.numInstances());
	Enumeration enumInsts = dataset.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    Instance softInstance;
	    if (instance instanceof SparseInstance)
		softInstance = new SoftClassifiedSparseInstance((SparseInstance)instance, randomizer);
	    else
		softInstance = new SoftClassifiedFullInstance(instance, randomizer);
	    m_Instances.addElement(softInstance);
	}
    }

    /**
     * Create a set of SoftClassifiedInstances from a given set of
     * Instances with hard class probabilities using existing class values.
   */
    public SoftClassifiedInstances (Instances dataset) {
	super(dataset, dataset.numInstances());
	Enumeration enumInsts = dataset.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    Instance softInstance;
	    if (instance instanceof SparseInstance)
		softInstance = new SoftClassifiedSparseInstance((SparseInstance)instance);
	    else
		softInstance = new SoftClassifiedFullInstance(instance);
	    m_Instances.addElement(softInstance);
	}
    }

    /** Add another set of instances to this set */
    public void addInstances (SoftClassifiedInstances instances) {
	Enumeration enumInsts = instances.enumerateInstances();
	while (enumInsts.hasMoreElements()) {
	    Instance instance = (Instance) enumInsts.nextElement();
	    m_Instances.addElement(instance);
	}
    }

}


    


