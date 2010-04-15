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
 *    AffineProbMetric.java
 *    Copyright (C) 2002-3 Mikhail Bilenko
 *
 */

package weka.deduping.metrics;

import java.text.*;
import java.io.*;
import java.util.*;
import weka.core.*;
import weka.deduping.*;
  

/** AffineProbMetric class implements a probabilistic model string edit distance with affine-cost gaps
 *
 *   @author Mikhail Bilenko<mbilenko@cs.utexas.edu>
 *   @version 1.1
 **/
public class AffineProbMetric extends StringMetric implements LearnableStringMetric, Serializable, OptionHandler {

  /* Current probabilities, log-probabilities and accumulated expectations
     for each edit operation */
  protected double [][] m_editopProbs;
  protected double [][] m_editopLogProbs;
  protected double [][] m_editopOccs;

  /** Parameters for the generative model */
  protected double m_noopProb, m_noopLogProb, m_noopOccs;  // matching
  protected double m_endAtSubProb, m_endAtSubLogProb, m_endAtSubOccs; // ending the alignment at M state
  protected double m_endAtGapProb, m_endAtGapLogProb, m_endAtGapOccs; // ending the alignment at D/I states
  protected double m_gapStartProb, m_gapStartLogProb, m_gapStartOccs; // starting a gap in the alignment
  protected double m_gapExtendProb, m_gapExtendLogProb, m_gapExtendOccs; // extending a gap in the alignment
  protected double m_gapEndProb, m_gapEndLogProb, m_gapEndOccs;  // ending a gap in the alignment
  protected double m_subProb, m_subLogProb, m_subOccs;  // continuing to match/substitute in state M

  /** parameters for the additive model, obtained from log-probs to speed up
      computations in the "testing" phase after weights have been learned */
  protected double [][] m_editopCosts; 
  protected double m_noopCost;
  protected double m_endAtSubCost; 
  protected double m_endAtGapCost; 
  protected double m_gapStartCost;
  protected double m_gapExtendCost;
  protected double m_gapEndCost;
  protected double m_subCost;

  /** true if we are using a generative model for distance in the "testing" phase after learning the parameters
      By default we want to use the additive model that uses probabilities converted to costs*/
  protected boolean m_useGenerativeModel = false;

  /** Maximum number of iterations for training the model; usually converge in <10 iterations */
  protected int m_numIterations = 20;

  /** Normalization of edit distance by string length; equivalent to
    using the posterior probability in the generative model*/
  protected boolean m_normalized = true;

  /** Minimal value of a probability parameter.  Particularly important when
   training sets are small to prevent zero probabilities. */
  protected double m_clampProb = 1e-5;

  /** A handy constant for insertions/deletions, we treat them as substitution with a null character */
  protected final char blank = 0;

  /** TODO:  given a corpus, populate this array with the characters that are actually encountered */
  protected char [] m_usedChars = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
				    'q','r','s','t','u','v','w','x','y','z',' ','!','\"','#','$','%',
				    '&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6',
				    '7','8','9',':',';','<','=','>','?','@','[','\\',']','^','_','`','{',
				    '|','}','~'};

  
  /** We can have different ways of converting from distance to similarity  */
  public static final int CONVERSION_LAPLACIAN = 1;
  public static final int CONVERSION_UNIT = 2;
  public static final int CONVERSION_EXPONENTIAL = 4;
  public static final Tag[] TAGS_CONVERSION = {
    new Tag(CONVERSION_UNIT, "similarity = 1-distance"),
    new Tag(CONVERSION_LAPLACIAN, "similarity=1/(1+distance)"),
    new Tag(CONVERSION_EXPONENTIAL, "similarity=exp(-distance)")
      };
  /** The method of converting, by default laplacian */
  protected int m_conversionType = CONVERSION_EXPONENTIAL;

  protected boolean m_verbose = false; 

  /**
   * set up an instance of AffineProbMetric
   */
  public AffineProbMetric () {
    m_editopProbs = new double[128][128];
    m_editopLogProbs = new double[128][128];
    m_editopOccs = new double[128][128];
    m_editopCosts = new double[128][128];
    initProbs();
    normalizeEmissionProbs();
    normalizeTransitionProbs();
    updateLogProbs();
    initCosts();
  }

  /**
   * Calculate the forward matrices
   * @param _s1 first string
   * @param _s2 second string
   * @return m_endAtSubProb*matrix[l1][l2][0] + m_endAtGapProb(matrix[l1][l2][1] +
   * matrix[l1][l2][2]) extendains the distance value
   */
  protected double[][][] forward (String _s1, String _s2) {
    char [] s1 = _s1.toCharArray();
    char [] s2 = _s2.toCharArray();
    int l1 = s1.length, l2 = s2.length;
    double matrix[][][] = new double[l1 + 1][l2 + 1][3];
    double tmpLog, subProb, tmpLog1;

    // initialization
    for (int i = 0; i <=l1; i++)
      matrix[i][0][0] = matrix[i][0][1] = matrix[i][0][2] = Double.NEGATIVE_INFINITY;
    for (int j = 1; j <=l2; j++)
      matrix[0][j][0] = matrix[0][j][1] = matrix[0][j][2] = Double.NEGATIVE_INFINITY;
    matrix[0][0][0] = 0;
	
    // border rows
    for (int j = 1; j <=l2; j++) {
      tmpLog = logSum(m_gapExtendLogProb + matrix[0][j-1][2], m_gapStartLogProb + matrix[0][j-1][0]); 
      matrix[0][j][2] = m_editopLogProbs[blank][s2[j-1]] + tmpLog;
    }
    for (int i = 1; i <= l1; i++) {
      tmpLog = logSum(m_gapStartLogProb + matrix[i-1][0][0], m_gapExtendLogProb + matrix[i-1][0][1]);
      matrix[i][0][1] = m_editopLogProbs[blank][s1[i-1]] + tmpLog;
    }
    
    // the rest
    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {
	tmpLog = logSum(m_gapStartLogProb + matrix[i-1][j][0], m_gapExtendLogProb + matrix[i-1][j][1]);
	matrix[i][j][1] = m_editopLogProbs[blank][s1[i-1]] + tmpLog;
	tmpLog = logSum(m_gapExtendLogProb + matrix[i][j-1][2], m_gapStartLogProb + matrix[i][j-1][0]);
	matrix[i][j][2] = m_editopLogProbs[blank][s2[j-1]] + tmpLog;
	subProb = ((s1[i-1] == s2[j-1]) ? m_noopLogProb : m_editopLogProbs[s1[i-1]][s2[j-1]]);
	tmpLog1 = logSum(m_subLogProb + matrix[i-1][j-1][0], m_gapEndLogProb + matrix[i-1][j-1][2]);
	tmpLog = logSum(m_gapEndLogProb + matrix[i-1][j-1][1], tmpLog1);
	matrix[i][j][0] =  subProb + tmpLog;
      }
    }
    return matrix;
  }

  /**
   * Calculate the backward matrices
   * @param _s1 first string
   * @param _s2 second string
   * @return matrix[0][0][0] extendains the distance value
   */
  protected double[][][] backward (String _s1, String _s2) {
    char [] s1 = _s1.toCharArray();
    char [] s2 = _s2.toCharArray();
    int l1 = s1.length, l2 = s2.length;
    double matrix[][][] = new double[l1 + 1][l2 + 1][3];
    double sub_pairProb, del_charProb, ins_charProb, tmpLog;

    // initialize
    for (int i = 0; i <=l1; i++)
      matrix[i][l2][0] = matrix[i][l2][1] = matrix[i][l2][2] = Double.NEGATIVE_INFINITY;
    for (int j = 0; j <=l2; j++)
      matrix[l1][j][0] = matrix[l1][j][1] = matrix[l1][j][2] = Double.NEGATIVE_INFINITY;
    matrix[l1][l2][0] = m_endAtSubLogProb;
    matrix[l1][l2][1] = matrix[l1][l2][2] = m_endAtGapLogProb;

    // border rows
    for (int i = l1-1; i >= 0; i--) {
      matrix[i][l2][0] = m_editopLogProbs[blank][s1[i]] + m_gapStartLogProb + matrix[i+1][l2][1];
      matrix[i][l2][1] = m_editopLogProbs[blank][s1[i]] + m_gapExtendLogProb + matrix[i+1][l2][1];
    }
    for (int j = l2-1; j >= 0; j--) {
      matrix[l1][j][0] = m_editopLogProbs[blank][s2[j]] + m_gapStartLogProb + matrix[l1][j+1][2];
      matrix[l1][j][2] = m_editopLogProbs[blank][s2[j]] + m_gapExtendLogProb + matrix[l1][j+1][2];
    }
    // fill the rest of the matrix
    for (int i = l1-1; i >= 0; i--) {
      for (int j = l2-1; j >= 0; j--) {
	ins_charProb = m_editopLogProbs[blank][s1[i]];
	del_charProb = m_editopLogProbs[blank][s2[j]];
	sub_pairProb = ((s1[i] == s2[j]) ? m_noopLogProb : m_editopLogProbs[s1[i]][s2[j]]);
	matrix[i][j][1] = logSum(ins_charProb + m_gapExtendLogProb + matrix[i+1][j][1],
				 sub_pairProb + m_gapEndLogProb + matrix[i+1][j+1][0]);
	matrix[i][j][2] = logSum(del_charProb + m_gapExtendLogProb + matrix[i][j+1][2],
				 sub_pairProb + m_gapEndLogProb + matrix[i+1][j+1][0]);
	tmpLog = logSum(ins_charProb + matrix[i+1][j][1], del_charProb + matrix[i][j+1][2]);
	matrix[i][j][0] = logSum(sub_pairProb + m_subLogProb + matrix[i+1][j+1][0],
				 m_gapStartLogProb + tmpLog);
      }
    }
    return matrix;
  }


    /**
   * print out the three matrices
   */
  public void printMatrices(String s1, String s2) {
    double[][][] forward = forward(s1, s2);
    double[][][] backward = backward(s1, s2);
    int l1 = s1.length(), l2 = s2.length();

    double totalForward = logSum(m_endAtSubLogProb + forward[l1][l2][0], m_endAtGapLogProb + forward[l1][l2][1]);
    totalForward = logSum(totalForward, m_endAtGapLogProb + forward[l1][l2][2]);
    System.out.println("\nB:" + backward[0][0][0] + "\tF:" + totalForward);
    
    System.out.println("\n***FORWARD***\nSUBSTITUTION:");
    printAlignmentMatrix(s1, s2, 0, forward);

    System.out.println("\n\nDELETION:");
    printAlignmentMatrix(s1, s2, 1, forward);

    System.out.println("\n\nINSERTION:");
    printAlignmentMatrix(s1, s2, 2, forward);


    System.out.println("\n***BACKWARD***\nSUBSTITUTION:");
    printAlignmentMatrix(s1, s2, 0, backward);

    System.out.println("\n\nDELETION:");
    printAlignmentMatrix(s1, s2, 1, backward);

    System.out.println("\n\nINSERTION:");
    printAlignmentMatrix(s1, s2, 2, backward);
  }

  public void printAlignmentMatrix(String _s1, String _s2, int idx, double[][][] matrix) {
    DecimalFormat fmt = new DecimalFormat ("0.000");
    char[] s1 = _s1.toCharArray();
    char[] s2 = _s1.toCharArray();
    
    System.out.print('\t');
    for (int i = 0; i < s2.length; i++) {
      System.out.print("\t" + s2[i]);
    }
    System.out.println();
    for (int i = 0; i < matrix.length; i++) {
      if (i > 0) System.out.print(s1[i-1] + "\t");  else System.out.print("\t");
      for (int j = 0; j < matrix[i].length; j++) {
	System.out.print(fmt.format(matrix[i][j][idx]) + "\t");
      }
      System.out.println();
    }
  } 
  
  /**
   * print out some data in case things go wrong
   */
  protected void printOpProbs() {
    System.out.println("extend_gap_op.prob=" + m_gapExtendProb +
		       "  end_gap_op.prob=" + m_gapEndProb +
		       "  subst_op.prob=" + m_subProb);
  }


        
  /**
   *  Train the distance parameters using provided examples using EM
   * @param matched_pairs Each member is a String[] extendaining two matching fields
   * @param matched_pairs Each member is a String[] extendaining two non-matching fields
   */ 
  public void trainMetric (ArrayList pairList) throws Exception {
    initProbs();
    recordCosts(0);

    // convert the training data to lower case
    for (int j = 0; j < pairList.size(); j++) {
      StringPair pair = (StringPair)pairList.get(j);
      pair.str1 = pair.str1.toLowerCase();
      pair.str2 = pair.str2.toLowerCase();
    }
	
    try {
      // dump out the current probablities
      PrintWriter out = new PrintWriter(new FileWriter("/tmp/probs1"));

      double totalProb = 0;
      double prevTotalProb = -Double.MAX_VALUE;
      for (int i = 1; i <= m_numIterations && Math.abs(totalProb - prevTotalProb) > 1; i++) {
	resetOccurrences();
	out.println(i + "\t" + m_endAtSubProb + "\t" + m_subProb + "\t" + m_gapStartProb +
		    "\t" + m_endAtGapProb + "\t" + m_gapEndProb + "\t" + m_gapExtendProb + "\t" + m_noopProb);

	// go through positives
	prevTotalProb = totalProb;
	totalProb = 0;
	for (int j = 0; j < pairList.size(); j++) {
	  StringPair pair = (StringPair)pairList.get(j);
	  if (pair.positive) {
	    totalProb += expectationStep (pair.str1, pair.str2, 1, true);
	  }
	}
	// go through negatives  - TODO - discriminative training
	//	    for (int j = 0; j < negExamples.length; j++)
	//		expectationStep (negExamples[j][1], negExamples[j][0], 1, false);

	System.out.println(i + ". Total likelihood=" + totalProb + ";  prev=" + prevTotalProb);
	System.out.println("************ Accumulated expectations ******************** ");
	System.out.println("End_s=" + m_endAtSubOccs + "\tSub=" + m_subOccs + "\tStGap=" + m_gapStartOccs +
			   "\nEnd_g=" + m_endAtGapOccs + "\tEndGap=" + m_gapEndOccs + " ContGap=" + m_gapExtendOccs +
			   "\nNoop=" + m_noopOccs);
	System.out.println("********************************");
	maximizationStep ();
      }
      out.close();
    } catch (Exception e) { e.printStackTrace();}
    initCosts();
    recordCosts(1);
  }
    
  /**
   * Expectation  part of the EM algorithm
   *  accumulates expectations of editop probabilities over example pairs
   *  Expectation is calculated based on two examples which are either duplicates (pos=true)
   *  or non-duplicates (pos=false).  Lambda is a weighting parameter, 1 by default.
   * @param _s1 first string
   * @param _s2 second string
   * @param lambda learning rate parameter, 1 by default
   * @param pos_training true if strings are matched, false if mismatched
   */
  protected double expectationStep (String _s1, String _s2, int lambda, boolean pos_training) {
    int l1 = _s1.length(), l2 = _s2.length();
    if (l1 == 0 || l2 == 0) {
      return 0;
    }
    char [] s1 = _s1.toCharArray();
    char [] s2 = _s2.toCharArray();
    double fMatrix[][][] = forward (_s1, _s2);
    double bMatrix[][][] = backward (_s1, _s2);
    double stringProb = bMatrix[0][0][0];
//  NB: b[0][0][0]must be equal to endAtSub*f[l1][l2][0] + endAtGap*(f[l1][l2][1]+f[l1][l2][2]); uncomment below for sanity check
//  double totalForward = logSum(m_endAtSubLogProb + fMatrix[l1][l2][0], m_endAtSubLogProb + fMatrix[l1][l2][1]);
//  totalForward = logSum(totalForward, m_endAtSubLogProb + fMatrix[l1][l2][2]);
//  System.out.println("b:" + bMatrix[0][0][0] + "\tf:" + totalForward);    

    double occsSubst, occsStartGap_1, occsStartGap_2, occsExtendGap_1, occsExtendGap_2;
    double occsEndGap_1, occsEndGap_2;
    double sub_pairProb, ins_charProb, del_charProb;
    char s1_i, s2_j;

    if (stringProb == 0.0) {
      System.out.println("TROUBLE!!!!  s1=" + _s1 + " s2=" + _s2);
      printMatrices(_s1,_s2);
      return 0;
    }
    m_endAtSubOccs += lambda; 
    m_endAtGapOccs += 2*lambda;

    for (int i = 1; i < l1; i++) {
      for (int j = 1; j< l2; j++) {
	s1_i = s1[i-1];
	s2_j = s2[j-1];
	if (s1_i == s2_j) {
	  sub_pairProb = m_noopLogProb;
	} else {
	  sub_pairProb = m_editopLogProbs[s1_i][s2_j];
	}
	ins_charProb = m_editopLogProbs[blank][s1_i]; 
	del_charProb = m_editopLogProbs[blank][s2_j];

	// substituting or matching
	occsSubst = Math.exp(fMatrix[i-1][j-1][0] + sub_pairProb + m_subLogProb + bMatrix[i][j][0] - stringProb);
	if (s1_i == s2_j) {
	  m_noopOccs += occsSubst;
	} else { 
	  m_editopOccs[s1_i][s2_j] +=occsSubst;
	}
	m_subOccs += occsSubst;
		
	// starting a gap
	occsStartGap_1 = Math.exp(fMatrix[i-1][j][0] + ins_charProb + m_gapStartLogProb + bMatrix[i][j][1]-stringProb);
	m_editopOccs[blank][s1_i] += occsStartGap_1;
	occsStartGap_2 = Math.exp(fMatrix[i][j-1][0] + del_charProb + m_gapStartLogProb + bMatrix[i][j][2]-stringProb);
	m_editopOccs[blank][s2_j] += occsStartGap_2;
	m_gapStartOccs += occsStartGap_1 + occsStartGap_2;
		
	// extendinuing a gap     
	occsExtendGap_1 = Math.exp(fMatrix[i-1][j][1] + ins_charProb + m_gapExtendLogProb + bMatrix[i][j][1]-stringProb);
	m_editopOccs[blank][s1_i] += occsExtendGap_1;
	occsExtendGap_2 = Math.exp(fMatrix[i][j-1][2] + del_charProb + m_gapExtendLogProb + bMatrix[i][j][2]-stringProb);
	m_editopOccs[blank][s2_j] += occsExtendGap_2;
	m_gapExtendOccs += occsExtendGap_1 + occsExtendGap_2;
		
	// ending a gap
	occsEndGap_1 = Math.exp(fMatrix[i-1][j-1][1] + sub_pairProb + m_gapEndLogProb + bMatrix[i][j][0] - stringProb);
	if (s1_i == s2_j) {
	  m_noopOccs += occsEndGap_1;
	} else { 
	  m_editopOccs[s1_i][s2_j] += occsEndGap_1;
	}
	occsEndGap_2 = Math.exp(fMatrix[i-1][j-1][2] + sub_pairProb + m_gapEndLogProb + bMatrix[i][j][0] - stringProb);
	if (s1_i == s2_j) {
	  m_noopOccs += occsEndGap_2;
	} else { 
	  m_editopOccs[s1_i][s2_j] += occsEndGap_2;
	}
	m_gapEndOccs += occsEndGap_1 + occsEndGap_2;
      }
    }
    // border rows.  We can't end gap, and can start/extend gap only one way
    for (int i = 1; i < l1; i++) {
      s1_i = s1[i-1];
      s2_j = s2[l2-1];
      ins_charProb = m_editopLogProbs[blank][s1_i];
      if (s1_i == s2_j) {
	sub_pairProb = m_noopLogProb;
      } else {
	sub_pairProb = m_editopLogProbs[s1_i][s2_j];
      }

      occsSubst = Math.exp(fMatrix[i-1][l2-1][0] + sub_pairProb + m_subLogProb + bMatrix[i][l2][0] - stringProb);
      if (s1_i == s2_j) {
	m_noopOccs += occsSubst;
      } else { 
	m_editopOccs[s1_i][s2_j] += occsSubst;
      }
      m_subOccs += occsSubst;
		
      occsStartGap_1 = Math.exp(fMatrix[i-1][l2][0] + ins_charProb + m_gapStartLogProb + bMatrix[i][l2][1] - stringProb);
      m_editopOccs[blank][s1_i] += occsStartGap_1;
      m_gapStartOccs += occsStartGap_1;

      occsExtendGap_1 = Math.exp(fMatrix[i-1][l2][1] + ins_charProb + m_gapExtendLogProb + bMatrix[i][l2][1] - stringProb);
      m_editopOccs[blank][s1_i] += occsExtendGap_1;
      m_gapExtendOccs += occsExtendGap_1;   //  DO WE NEED THIS??? WE HAD NO CHOICE!
    }
    for (int j = 1; j < l2; j++) {
      s1_i = s1[l1-1];
      s2_j = s2[j-1];
      del_charProb = m_editopLogProbs[blank][s2_j];
      if (s1_i == s2_j) {
	sub_pairProb = m_noopLogProb;
      } else {
	sub_pairProb = m_editopLogProbs[s1_i][s2_j];
      }

      occsSubst = Math.exp(fMatrix[l1-1][j-1][0] + sub_pairProb + m_subLogProb + bMatrix[l1][j][0] - stringProb);
      if (s1_i == s2_j) {
	m_noopOccs += occsSubst;
      } else { 
	m_editopOccs[s1_i][s2_j] += occsSubst;
      }
      m_subOccs += occsSubst;
	    
      occsStartGap_2 = Math.exp(fMatrix[l1][j-1][0] + del_charProb + m_gapStartLogProb + bMatrix[l1][j][2] - stringProb);
      m_editopOccs[blank][s2_j] += occsStartGap_2;
      m_gapStartOccs += occsStartGap_2;
	    
      occsExtendGap_2 = Math.exp(fMatrix[l1][j-1][2] + del_charProb + m_gapExtendLogProb + bMatrix[l1][j][2] - stringProb);
      m_editopOccs[blank][s2_j] += occsExtendGap_2;
      m_gapExtendOccs += occsExtendGap_2;  //   DO WE NEED THIS??? WE HAD NO CHOICE!
    }
    return stringProb;
  }

  /** 
   *  Maximization step of the EM algorithm
   */
  protected void maximizationStep () {
    double N, N_s, N_id;

    // TODO:  when trying to incorporate discriminative training, see EditDistance.java
    // in old codebase for an earlier attempt to deal with negative expectations

    // Sum up expectations for transitions in substitution state
    N = m_subOccs + 2*m_gapStartOccs + m_endAtSubOccs;
    m_subProb = m_subOccs / N;
    m_gapStartProb = m_gapStartOccs / N;
    m_endAtSubProb = m_endAtSubOccs / N;
    if (m_subProb < m_clampProb) m_subProb = m_clampProb;
    if (m_gapStartProb < m_clampProb) m_gapStartProb = m_clampProb;
    if (m_endAtSubProb < m_clampProb) m_endAtSubProb = m_clampProb;

    // Sum up expectations for occurrences in deletion/insertion states
    N = m_gapExtendOccs + m_gapEndOccs + m_endAtGapOccs;
    m_gapExtendProb = m_gapExtendOccs / N;
    m_gapEndProb = m_gapEndOccs / N;
    m_endAtGapProb = m_endAtGapOccs / N;
    if (m_gapExtendProb < m_clampProb) m_gapExtendProb = m_clampProb;
    if (m_gapEndProb < m_gapEndProb) m_gapEndProb = m_clampProb;
    if (m_endAtGapProb < m_endAtGapProb) m_endAtGapProb = m_clampProb;
	
    // Now let's add up expectations for actual edit operators
    // we add up separately for substitution and deletion/insertion
    N_s = m_noopOccs;
    N_id = 0;
    for (int i = 0; i < m_usedChars.length; i++) {
      N_id += m_editopOccs[blank][m_usedChars[i]];
      for (int j = 0; j < m_usedChars.length; j++) {
	if (i != j) {
	  N_s += m_editopOccs[m_usedChars[i]][m_usedChars[j]];
	} 
      }
    }

    // Recalculate the probabilities
    m_noopProb = m_noopOccs / N_s;
    for (int i = 0; i < m_usedChars.length; i++) {
      m_editopProbs[blank][m_usedChars[i]] = Math.max(m_clampProb, m_editopOccs[blank][m_usedChars[i]] / N_id);
      for (int j = i+1; j < m_usedChars.length; j++) {
	m_editopProbs[m_usedChars[i]][m_usedChars[j]] = Math.max(m_clampProb, 
	  (m_editopOccs[m_usedChars[i]][m_usedChars[j]] + m_editopOccs[m_usedChars[j]][m_usedChars[i]])/ N_s);
	m_editopProbs[m_usedChars[j]][m_usedChars[i]] = m_editopProbs[m_usedChars[i]][m_usedChars[j]];
      }
    }
    normalizeTransitionProbs();
    normalizeEmissionProbs();
    updateLogProbs();
  }
    

  /** 
   * Normalize the probabilities of emission editops so that they sum to 1
   * for each state
   */
  protected void normalizeEmissionProbs() {
    double N_s = m_noopProb, N_id = 0;
    for (int i = 0; i < m_usedChars.length; i++) {
      N_id += m_editopProbs[blank][m_usedChars[i]];
      for (int j = i+1; j < m_usedChars.length; j++) {
	N_s += m_editopProbs[m_usedChars[i]][m_usedChars[j]];
      }
    }
    // Recalculate the probabilities
    m_noopProb = m_noopProb / N_s;
    for (int i = 0; i < m_usedChars.length; i++) {
      m_editopProbs[blank][m_usedChars[i]] /= N_id;
      for (int j = i+1; j < m_usedChars.length; j++) {
	m_editopProbs[m_usedChars[i]][m_usedChars[j]] /= N_s;
	m_editopProbs[m_usedChars[j]][m_usedChars[i]] = m_editopProbs[m_usedChars[i]][m_usedChars[j]];
      }
    }
  }

  /** 
   * Normalize the probabilities of transitions so that they sum to 1
   * for each state
   */
  protected void normalizeTransitionProbs() {
    double P;

    // M-state
    P = m_subProb + 2 * m_gapStartProb + m_endAtSubProb;
    m_subProb /= P;
    m_gapStartProb /= P;
    m_endAtSubProb /= P;

    // I/D states
    P = m_gapExtendProb + m_gapEndProb + m_endAtGapProb;
    m_gapExtendProb /= P;
    m_gapEndProb /= P;
    m_endAtGapProb /= P;
  }
    
  /** 
   *  reset the number of occurrences of all ops in the set
   */
  protected void resetOccurrences () {
    m_noopOccs = 0;
    m_endAtSubOccs = 0;
    m_endAtGapOccs = 0;
    m_gapStartOccs = 0;
    m_gapExtendOccs = 0;
    m_gapEndOccs = 0;
    m_subOccs = 0;
    for (int i = 0; i < m_usedChars.length; i++) {
      m_editopOccs[blank][m_usedChars[i]] = 0;
      for (int j = 0; j < m_usedChars.length; j++) {
	m_editopOccs[m_usedChars[i]][m_usedChars[j]] = 0;
      }
    }
  }

  /** 
   *  initialize the probabilities to some startup values
   */
  protected void initProbs () {
    double probDeletionUniform = 1.0 / m_usedChars.length;
    double probSubUniform = probDeletionUniform * probDeletionUniform;
    m_endAtSubProb = 0.05;
    m_endAtGapProb = 0.1;
    m_gapStartProb = 0.05;
    m_gapExtendProb = 0.5;
    m_gapEndProb = 0.4;
    m_subProb = 0.9;
    m_noopProb = 0.9;
//      m_endAtSubProb = 0.1;
//      m_endAtGapProb = 0.1;
//      m_gapStartProb = 0.4;
//      m_gapExtendProb = 0.6;
//      m_gapEndProb = 0.3;
//      m_subProb = 0.5;
//      m_noopProb = 0.3;
    for (int i = 0; i < m_usedChars.length; i++) {
      m_editopProbs[blank][m_usedChars[i]] = probDeletionUniform;
      for (int j = 0; j < m_usedChars.length; j++) {
	m_editopProbs[m_usedChars[i]][m_usedChars[j]] = probSubUniform;
      }
    }
  }

  /** 
   *  initialize the costs using current values of the probabilities
   */
  protected void initCosts () {
    m_gapStartCost =  -m_gapStartLogProb;
    m_gapExtendCost =   -m_gapExtendLogProb;
    m_endAtSubCost = -m_endAtSubLogProb;
    m_endAtGapCost = -m_endAtGapLogProb;
    m_gapEndCost =    -m_gapEndLogProb;
    m_subCost =        -m_subLogProb;
    m_noopCost =     -m_noopLogProb;

    if (m_verbose) {
      System.out.println("\nScaled by extend cost:\nGapStrt=" + (m_gapStartCost/m_gapExtendCost) +
			 "\tGapExt=" + (m_gapExtendCost/m_gapExtendCost) +
			 "\tGapEnd=" + (m_gapEndCost/m_gapExtendCost) + 
			 "\tSub=" + (m_subCost/m_gapExtendCost) +
			 "\tNoop=" + (m_noopCost/m_gapExtendCost));
      System.out.println("\nActual costs:\nGapStrt=" + (m_gapStartCost) +
			 "\tGapExt=" + (m_gapExtendCost) +
			 "\tGapEnd=" + (m_gapEndCost) + 
			 "\tSub=" + (m_subCost) +
			 "\tNoop=" + (m_noopCost));

    }
    if (!m_useGenerativeModel) {
      for (int i = 0; i < m_usedChars.length; i++) {
	m_editopCosts[blank][m_usedChars[i]] =   -m_editopLogProbs[blank][m_usedChars[i]];
	for (int j = 0; j < m_usedChars.length; j++) {
	  m_editopCosts[m_usedChars[i]][m_usedChars[j]] =
	    -m_editopLogProbs[m_usedChars[i]][m_usedChars[j]];
	}
      }
    } else {
      m_editopCosts = new double[128][128];
    }
  }

  /**
   * store logs of all probabilities in m_editopLogProbs
   */
  protected void updateLogProbs() {
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 128; j++) {
	m_editopLogProbs[i][j] = (m_editopProbs[i][j] == 0) ? -Double.MAX_VALUE : Math.log(m_editopProbs[i][j]);
      }
    }
    m_noopLogProb = Math.log(m_noopProb);
    m_endAtSubLogProb = Math.log(m_endAtSubProb);
    m_endAtGapLogProb = Math.log(m_endAtGapProb);
    m_gapStartLogProb = Math.log(m_gapStartProb);
    m_gapExtendLogProb = Math.log(m_gapExtendProb);
    m_gapEndLogProb = Math.log(m_gapEndProb);
    m_subLogProb = Math.log(m_subProb);

    DecimalFormat fmt = new DecimalFormat ("0.0000");
    System.out.println("After update:\tNOOP=" + fmt.format(m_noopProb) + "=" + fmt.format(m_noopLogProb) +
		       "\tSUB=" + fmt.format(m_subProb) + "=" + fmt.format(m_subLogProb) +
		       "\n\t\tGAPst=" + fmt.format(m_gapStartProb) + "=" + fmt.format(m_gapStartLogProb) +
		       "\tGAPcont=" + fmt.format(m_gapExtendProb) + "=" + fmt.format(m_gapExtendLogProb) +
		       "\tGAPend=" + fmt.format(m_gapEndProb) + "=" + fmt.format(m_gapEndLogProb) +
		       "\n\t\tendAtGap=" + fmt.format(m_endAtGapProb) + "=" + fmt.format(m_endAtGapLogProb) +
		       "\tendAtSub=" + fmt.format(m_endAtSubProb) + "=" + fmt.format(m_endAtSubLogProb));
  }

    
  /** 
   * Get the distance between two strings
   * @param s1 first string
   * @param s2 second string
   * @return a value of this distance between these two strings
   */
  public double distance (String s1, String s2) {
    if (m_useGenerativeModel) {
      double d = backward(s1,s2)[0][0][0];
      if (m_normalized) {
	//	for (int i = 0; i < (s1.length() + s2.length()); i++) 
	  // TODO:  fix the posteriors; don't care for now - we always use the additive model
	//	  d -= m_noopLogProb + m_subLogProb;
	//  		for (int i = 0; i < s1.length(); i++) {
	//  		    d -= m_editopLogProbs[blank][s1.charAt(i)];
	//  		}
	//  		for (int i = 0; i < s2.length(); i++) {
	//  		    d -= m_editopLogProbs[blank][s2.charAt(i)];
	//  		}
      }
      return -d;
    } else {
      return costDistance(s1, s2);
    }
  }


  /** Method:  recordCosts
      Record probability matrix for further MatLab use
  */
  void recordCosts(int id) {
    try {
      FileOutputStream fstr = new FileOutputStream ("/tmp/probs/ProbAffineCosts.txt", true);
      DataOutputStream out = new DataOutputStream (fstr);
      char s, t;
      DecimalFormat fmt = new DecimalFormat ("0.00");
	    
      out.writeBytes(id + " " + m_noopCost + "\n" + "char\tblank   ");
      for (int i = 0; i < m_usedChars.length; i++) {
	out.writeBytes("\'" + m_usedChars[i] + "\'" + "\t");
      }
      out.writeBytes("\n");
      for (int i = 0; i < m_usedChars.length; i++) {
	out.writeBytes("\'" + m_usedChars[i] + "\'" + "\t" +
		       fmt.format(m_editopCosts[blank][m_usedChars[i]]) + "\t"); 
	for (int j = 0; j < i; j++) {
	  out.writeBytes(fmt.format(m_editopCosts[m_usedChars[i]][m_usedChars[j]]) + "\t");
	}
	out.writeBytes("\n");
      }
      out.close();
    } catch (Exception x) {}
  }
	    

  static String MatrixToString (double matrix[][]) {
    DecimalFormat fmt = new DecimalFormat ("0.00");
    String s = "";
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++)
	s = s + fmt.format(matrix[i][j]) + "  ";
      s = s + "\n";
    }
    return s;
  }

  static String intMatrixToString (int matrix[][]) {
    String s = "";
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++)
	s = s + matrix[i][j] + "  ";
      s = s + "\n";
    }
    return s;
  }

  static String doubleMatrixToString (double matrix[][]) {
    String s = "";
    java.text.DecimalFormat de = new java.text.DecimalFormat("0.0E000");
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++)
	s = s + de.format(matrix[i][j]) + "  ";
      s = s + "\n";
    }
    return s;
  }

  static String doubleMatrixToString0 (double matrix[][][], int k) {
    String s;
    s = "";
    java.text.DecimalFormat de = new java.text.DecimalFormat("0.0E000");
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++)
	s = s + de.format(matrix[i][j][k]) + "  ";
      s = s + "\n";
    }
    return s;
  }

  static String charMatrixToString (char matrix[][]) {
    String s = "";
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++)
	s = s + matrix[i][j] + "  ";
      s = s + "\n";
    }
    return s;
  }


  /** Calculation of log(a+b) with a correction for machine precision
   * @param _a number log(a)
   * @param _b number log(b)
   * @returns log(a+b)
   */
  protected double logSum(double _logA, double _logB) {
    double logSum = 0;
    // make logA the smaller of the two
    double logA = (_logA < _logB) ? _logA : _logB;
    double logB = (_logA < _logB) ? _logB : _logA;
    
    if (logA - logB < -324 || logA == Double.NEGATIVE_INFINITY) {
      logSum = logB;
    } else {
      logSum = logA + Math.log(1 + Math.exp(logB - logA));
    }
    return logSum;
  }

  /**
   * Calculate affine gapped distance using learned costs
   * @param s1 first string
   * @param s2 second string
   * @return minimum number of deletions/insertions/substitutions to be performed
   * to transform s1 into s2 (or vice versa)
   */
  public double costDistance(String string1, String string2) {
    char [] s1 = string1.toLowerCase().toCharArray();
    char [] s2 = string2.toLowerCase().toCharArray();
    int l1 = s1.length, l2 = s2.length;
    double T[][] = new double[l1+1][l2+1];
    double I[][] = new double[l1+1][l2+1];
    double D[][] = new double[l1+1][l2+1];
    double subCost = 0, sub_charCost = 0, ins_charCost = 0, del_charCost = 0, ret;
    int i, j;

    if (l1==0 || l2==0) {
      return m_gapStartCost + (l1+l2-1) * m_gapExtendCost;
    }
    for (j = 0; j < l2+1; j++) {
      I[0][j] = Double.MAX_VALUE;
      D[0][j] = Double.MAX_VALUE;
    }
    for (j = 0; j < l1+1; j++) {
      I[j][0] = Double.MAX_VALUE;
      D[j][0] = Double.MAX_VALUE;
    }
    T[0][0] = 0;
    T[0][1] = m_gapStartCost;
    T[1][0] = m_gapStartCost;
    for (j = 2; j < l2+1; j++) {
      ins_charCost = m_editopCosts[blank][s2[j-1]];
      T[0][j] = T[0][j-1] + m_gapExtendCost + ins_charCost;
    }
    for (j = 2; j < l1+1; j++) {
      del_charCost =  m_editopCosts[blank][s1[j-1]];
      T[j][0] = T[j-1][0] + m_gapExtendCost + del_charCost;
    }
    for (i = 1; i < l1+1; i++) {
      for (j = 1; j < l2+1; j++) {
	char c1 = s1[i-1];
	char c2 = s2[j-1];
	del_charCost = m_editopCosts[blank][c1];
	ins_charCost = m_editopCosts[blank][c2];
	sub_charCost = (c1 == c2) ? m_noopCost : m_editopCosts[c1][c2];
	sub_charCost = (c1 == c2) ? 0 : m_editopCosts[c1][c2];  //  ??  do we use noopCost?

	if (D[i-1][j]+m_gapExtendCost > T[i-1][j]+m_gapStartCost) {
	  D[i][j] = T[i-1][j]+m_gapStartCost + del_charCost;
	} else {
	  D[i][j] = D[i-1][j]+m_gapExtendCost + del_charCost;
	}
		
	if (I[i][j-1]+m_gapExtendCost > T[i][j-1]+m_gapStartCost) {
	  I[i][j] = T[i][j-1] + m_gapStartCost + ins_charCost;
	} else {
	  I[i][j] = I[i][j-1] + m_gapExtendCost + ins_charCost;
	}
		
	//subCost = m_subCost + sub_charCost;
	subCost =((c1 == c2) ? 0 : (m_subCost + m_gapEndCost));
	subCost = subCost + sub_charCost;
  	subCost = (c1 == c2) ? 0 : m_subCost;
	//	subCost = m_subCost;
		
	if  ((T[i-1][j-1] + subCost < D[i-1][j-1] + m_gapEndCost) &&    /// d[i][j] or d[i-1][j-1]??
	     (T[i-1][j-1] + subCost < I[i-1][j-1] + m_gapEndCost )) {
	  T[i][j] = T[i-1][j-1] + subCost + sub_charCost;   // ?? do we add subCharCost?
	} else {
	  if (D[i-1][j-1] < I[i-1][j-1]) {
	    T[i][j] = D[i-1][j-1] + m_gapEndCost + sub_charCost;
	  } else {
	    T[i][j] = I[i-1][j-1] + m_gapEndCost + sub_charCost;
	  }
	}
      }
    }
	
    if (T[l1][l2] < D[l1][l2] && T[l1][l2] < I[l1][l2]) {
      ret = T[l1][l2];
    } else if (D[l1][l2] < I[l1][l2]) {
      ret = D[l1][l2];
    } else {
      ret = I[l1][l2];
    }
    if (m_normalized) {
//        // get the normalization factor as P(x,y)=P(x)P(y)
//        double Pxy = 2 * m_gapStartCost; 
//        for (int k = 0; k < l1; k++) {
//  	Pxy += s1[k] + m_gapExtendCost;
//        }
//        for (int k = 0; k < l2; k++) {
//  	Pxy += s2[k] + m_gapExtendCost;
//        }
//        ret /= Pxy;
      ret /= 4*(l1 + l2);
    }
    return ret;
  }

  public static void print3dMatrix(double [][][] matrix) {
    DecimalFormat fmt = new DecimalFormat ("0.0000E00");
    for (int i = 0; i < matrix[0][0].length; i++) {
      System.out.println ("\nMatrix[][][" + i + "]");
      for (int j = 0; j < matrix[0].length; j++) {
	for (int k = 0; k < matrix.length; k++) {
	  System.out.print(fmt.format(matrix[k][j][i]) + "\t");
	}
	System.out.println();
      }
    }
  }

  /** Set the distance to be normalized by the sum of the string's lengths
   * @param normalized if true, distance is normalized by the sum of string's lengths
   */
  public void setNormalized(boolean normalized) {
    m_normalized = normalized;
  } 

  /** Get whether the distance is normalized by the sum of the string's lengths
   * @return if true, distance is normalized by the sum of string's lengths
   */
  public boolean getNormalized() {
    return m_normalized;
  }


  /** Set the distance to use the generative model or convert back to the additive model
   * @param useGenerativeModel if true, the generative model is used
   */
  public void setUseGenerativeModel(boolean useGenerativeModel) {
    m_useGenerativeModel = useGenerativeModel;
  } 


  /** Do we use the generative model or convert back to the additive model?
   * @param useGenerativeModel if true, the generative model is used
   */
  public boolean getUseGenerativeModel() {
    return m_useGenerativeModel;
  } 

  /** Set the clamping probability value
   * @param clampProb a lower bound for all probability values to prevent underflow
   */
  public void setClampProb(double clampProb) {
    m_clampProb = clampProb;
  }
  
  /** Get the clamping probability value
   * @return a lower bound for all probability values to prevent underflow
   */
  public double getClampProb() {
    return m_clampProb;
  }

  /** Set the number of training iterations
   * @param numIterations the number of iterations
   */
  public void setNumIterations(int numIterations) {
    m_numIterations = numIterations;
  } 

  /** Get the number of training iterations
   * @return the number of training iterations
   */
  public int setNumIterations() {
    return m_numIterations;
  } 


  /** Create a copy of this metric
   * @return another AffineMetric with the same exact parameters as this  metric
   */
  public Object clone() {
    AffineProbMetric metric = new AffineProbMetric();
    metric.setNormalized(m_normalized);
    metric.setUseGenerativeModel(m_useGenerativeModel);
    metric.setClampProb(m_clampProb);
    metric.setNumIterations(m_numIterations);
    return metric;
  }


    /**
   * Gets the current settings of WeightedDotP.
   *
   * @return an array of strings suitable for passing to setOptions()
   * TODO!!!!
   */
  public String [] getOptions() {
    String [] options = new String [10];
    int current = 0;

    if (m_normalized) {
      options[current++] = "-N";
    }

    if (m_useGenerativeModel) {
      options[current++] = "-G";
    } else {
      options[current++] = "-A";
    }

    options[current++] = "-c";
    options[current++] = "" + m_clampProb;

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }


  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -N normalize by length
   * -m matchCost
   * -s subCost
   * -g gapStartCost
   * -e gapExtendCost   
   */
  public void setOptions(String[] options) throws Exception {
    setNormalized(Utils.getFlag('N', options));
   
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   * TODO!!!
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector(5);

    newVector.addElement(new Option("\tNormalize by lengths\n",
				    "N", 0, "-N"));

    
    return newVector.elements();
  }
  
  

  /** The computation of a metric can be either based on distance, or on similarity
   * @returns true
   */
  public boolean isDistanceBased() {
    return true;
  }


  /**
   * Returns a similarity estimate between two strings. Similarity is obtained by
   * inverting the distance value using one of three methods:
   * CONVERSION_LAPLACIAN, CONVERSION_EXPONENTIAL, CONVERSION_UNIT.
   * @param string1 First string.
   * @param string2 Second string.
   * @exception Exception if similarity could not be estimated.
   */
  public double similarity(String string1, String string2) throws Exception {
    switch (m_conversionType) {
    case CONVERSION_LAPLACIAN: 
      return 1 / (1 + distance(string1, string2));
    case CONVERSION_UNIT:
      return 2 * (1 - distance(string1, string2));
    case CONVERSION_EXPONENTIAL:
      return Math.exp(-distance(string1, string2));
    default:
      throw new Exception ("Unknown distance to similarity conversion method");
    }
  }

  /**
   * Set the type of similarity to distance conversion. Values other
   * than CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL will be ignored
   * 
   * @param type type of the similarity to distance conversion to use
   */
  public void setConversionType(SelectedTag conversionType) {
    if (conversionType.getTags() == TAGS_CONVERSION) {
      m_conversionType = conversionType.getSelectedTag().getID();
    }
  }

  /**
   * return the type of similarity to distance conversion
   * @return one of CONVERSION_LAPLACIAN, CONVERSION_UNIT, or CONVERSION_EXPONENTIAL
   */
  public SelectedTag getConversionType() {
    return new SelectedTag(m_conversionType, TAGS_CONVERSION);
  }


    
  public static void main(String[] args) {
    try { 
    AffineProbMetric metric = new AffineProbMetric();
    //    metric.trainMetric(new ArrayList());
    String s1 = new String("abcde");
    String s2 = new String("ab");
    metric.printMatrices(s1, s2);
    } catch (Exception e) { e.printStackTrace();}
  }
    
}
