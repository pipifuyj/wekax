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
 *    StringMetric.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */


package weka.deduping.metrics;

import weka.core.Utils;

/**
 * An abstract class that returns a measure of similarity between strings
 *
 * @author Mikhail Bilenko
 */

public abstract class StringMetric implements Cloneable {

  /** Compute a measure of distance between two strings
   * @param s1 first string
   * @param s2 second string
   * @returns distance between two strings
   */
  public abstract double distance(String s1, String s2) throws Exception;

  /** Compute a measure of similarity between two strings
   * @param s1 first string
   * @param s2 second string
   * @returns similarity between two strings
   */
  public abstract double similarity(String s1, String s2) throws Exception;


  /** The computation of a metric can be either based on distance, or on similarity
   * @returns true if the underlying metric computes distance, false if similarity
   */
  public abstract boolean isDistanceBased();
  
  /** Create a copy of this metric */
  public abstract Object clone();

  /**
   * Creates a new instance of a metric given it's class name and
   * (optional) arguments to pass to it's setOptions method. If the
   * classifier implements OptionHandler and the options parameter is
   * non-null, the classifier will have it's options set.
   *
   * @param metricName the fully qualified class name of the metric 
   * @param options an array of options suitable for passing to setOptions. May
   * be null.
   * @return the newly created metric ready for use.
   * @exception Exception if the metric  name is invalid, or the options
   * supplied are not acceptable to the metric 
   */
  public static StringMetric forName(String metricName,
				     String [] options) throws Exception {
    return (StringMetric)Utils.forName(StringMetric.class,
				       metricName,
				       options);
  }
      
}

