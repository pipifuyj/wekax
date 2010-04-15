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
 *    LearnableStringMetric.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */


package weka.deduping.metrics;


import weka.core.Instances;
import java.util.ArrayList;
/**
 * An interface for learnable string metrics
 *
 * @author Mikhail Bilenko
 */

public interface LearnableStringMetric {

  /** Train a metric given a set of aligned strings
   * @param pairList the training data as a list of StringPair's
   * @returns distance between two strings
   */
  public abstract void trainMetric(ArrayList pairList) throws Exception;


}

