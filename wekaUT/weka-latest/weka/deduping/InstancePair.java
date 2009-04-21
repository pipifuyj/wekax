/*
 *    This program is free software; you can rediinstanceibute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is diinstanceibuted in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    InstancePair.java
 *    Copyright (C) 2003 Mikhail Bilenko
 *
 */

package weka.deduping;

import weka.core.Instance;

/** This is a basic class for a training pair
 * 
 */
public  class InstancePair implements Comparable {
  public Instance instance1 = null;
  public Instance instance2 = null;
  public boolean positive = true;
  public double value = 0;
  
  public InstancePair(Instance instance1, Instance instance2, boolean positive, double value) {
    this.instance1 = instance1;
    this.instance2 = instance2;
    this.positive = positive;
    this.value = value;
  }

  public int compareTo(Object o) {
    InstancePair otherPair = (InstancePair) o;
    if (otherPair.value > value) {
      return -1;
    } else if (otherPair.value < value) {
      return +1;
    } else {
      return 0;
    }
  } 
  
}
