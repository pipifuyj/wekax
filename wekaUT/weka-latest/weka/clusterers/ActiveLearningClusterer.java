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
 *    ActiveLearningClusterer.java
 *    Copyright (C) 2001  Sugato Basu
 *
 */


/**
 * Active-learning Clusterer interface.
 */

package weka.clusterers;

public interface ActiveLearningClusterer {
  /** Returns the list of best instances for active learning */
  abstract int[] bestInstancesForActiveLearning(int num) throws Exception;

 /** Returns the list of best pairs for active learning */
  abstract InstancePair[] bestPairsForActiveLearning(int num) throws Exception;
}
