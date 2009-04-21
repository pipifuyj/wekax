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
 *    InstanceReference.java
 *    Copyright (C) 2003 Mikhail Bilenko, Raymond J. Mooney
 *
 */


package weka.deduping.blocking;

import java.util.*;
import java.io.Serializable;
import weka.core.*;
import weka.deduping.metrics.*;

/**
 * This class describes a basic data structure for storing a reference
 * to a database record.  This is used in the inverted index of
 * Blocking.  Structure is largely borrowed from StringReference.
 *
 * @author Mikhail Bilenko */


import java.io.*;

public class InstanceReference {
  /** The referenced instance. */
  public Instance instance = null;

  /** The index of the instance in the dataset */
  public int idx = -1; 

  /** The amalgamated instance string */
  public String string = null;
  
  /** The corresponding HashMapVector */
  public HashMapVector vector = null;

  /** The length of the corresponding instance vector. */
  public double length = 0.0;

  public InstanceReference(Instance instance, int index, String string, HashMapVector vector, double length) {
    this.instance = instance;
    this.idx = index; 
    this.string = string;
    this.vector = vector;
    this.length = length;
  }

  /** Create a reference to this record, initializing its length to 0 */
  public InstanceReference(Instance instance, int index, String string, HashMapVector vector) {
    this(instance, index, string, vector, 0.0);
  }

  public String toString() {
    return string;
  }
}









