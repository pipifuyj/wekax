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
 *    LearnableTokenEDAffine.java
 *    Copyright (C) 2002-3 Mikhail Bilenko
 *
 */

package weka.deduping.metrics;

import java.text.*;
import java.io.*;
import java.util.*;
import weka.core.*;
import weka.deduping.*;
  

/** LearnableTokenEDAffine class implements a probabilistic model string edit distance with affine-cost gaps
 *
 *   @author Mikhail Bilenko<mbilenko@cs.utexas.edu>
 *   @version 1.1
 **/
public class LearnableTokenEDAffine extends StringMetric implements LearnableStringMetric, Serializable, OptionHandler {

  /** The tokenizer */
  protected Tokenizer m_tokenizer = new WordTokenizer();

  /** A hashmap where observed strings are mapped to their TokenString representations */
  protected HashMap m_stringTokenStringMap = new HashMap();

  /** Parameters for the generative model */
  protected double m_subProb, m_subLogProb, m_subOccs;                           // continuing to match/substitute in state M
  protected double m_endAtSubProb, m_endAtSubLogProb, m_endAtSubOccs;            // ending the alignment at M state
  protected double m_endAtGapProb, m_endAtGapLogProb, m_endAtGapOccs;            // ending the alignment at D/I states
  protected double m_gapStartProb, m_gapStartLogProb, m_gapStartOccs;            // starting a gap in the alignment
  protected double m_gapExtendProb, m_gapExtendLogProb, m_gapExtendOccs;         // extending a gap in the alignment
  protected double m_gapEndProb, m_gapEndLogProb, m_gapEndOccs;                  // ending a gap in the alignment

  /** Emission probs */
  protected double m_matchProb, m_matchLogProb, m_matchOccs;
  protected double m_nonMatchProb, m_nonMatchLogProb, m_nonMatchOccs;
  protected double m_gapTokenProb, m_gapTokenLogProb, m_gapTokenOccs;

  /** parameters for the additive model, obtained from log-probs to speed up
      computations in the "testing" phase after weights have been learned */
  protected double m_matchCost, m_nonMatchCost;
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
   * set up an instance of LearnableTokenEDAffine
   */
  public LearnableTokenEDAffine () {
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
  protected double[][][] forward (TokenString ts1, TokenString ts2) {
    int [] s1 = ts1.tokenIDs;
    int [] s2 = ts2.tokenIDs;
    int l1 = s1.length, l2 = s2.length;
    double matrix[][][] = new double[l1 + 1][l2 + 1][3];
    double tmpLog, subProb, tmpLog1;

    // initialization - first the substitution/matching matrix
    for (int i = 0; i <=l1; i++) { 
      matrix[i][0][0] = matrix[i][0][1] = matrix[i][0][2] = Double.NEGATIVE_INFINITY;
    }
    for (int j = 1; j <=l2; j++) { 
      matrix[0][j][0] = matrix[0][j][1] = matrix[0][j][2] = Double.NEGATIVE_INFINITY;
    }
    matrix[0][0][0] = 0;
	
    // border rows in insertion/deletion matrices
    for (int i = 1; i <= l1; i++) {
      matrix[i][0][1] = m_gapTokenLogProb + logSum(m_gapExtendLogProb + matrix[i-1][0][1], m_gapStartLogProb + matrix[i-1][0][0]);
    }
    for (int j = 1; j <=l2; j++) {
      matrix[0][j][2] = m_gapTokenLogProb + logSum(m_gapExtendLogProb + matrix[0][j-1][2], m_gapStartLogProb + matrix[0][j-1][0]); 
    }
    
    // fill the matrices
    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {
	matrix[i][j][1] = m_gapTokenLogProb + logSum(m_gapStartLogProb + matrix[i-1][j][0], m_gapExtendLogProb + matrix[i-1][j][1]);
	matrix[i][j][2] = m_gapTokenLogProb + logSum(m_gapStartLogProb + matrix[i][j-1][0], m_gapExtendLogProb + matrix[i][j-1][2]);
	
	subProb = (s1[i-1] == s2[j-1]) ? m_matchLogProb : m_nonMatchLogProb;
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
  protected double[][][] backward (TokenString ts1, TokenString ts2) {
    int [] s1 = ts1.tokenIDs;
    int [] s2 = ts2.tokenIDs;
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
      matrix[i][l2][0] = m_gapTokenLogProb + m_gapStartLogProb + matrix[i+1][l2][1];
      matrix[i][l2][1] = m_gapTokenLogProb + m_gapExtendLogProb + matrix[i+1][l2][1];
    }
    for (int j = l2-1; j >= 0; j--) {
      matrix[l1][j][0] = m_gapTokenLogProb + m_gapStartLogProb + matrix[l1][j+1][2];
      matrix[l1][j][2] = m_gapTokenLogProb + m_gapExtendLogProb + matrix[l1][j+1][2];
    }
    
    // fill the rest of the matrix
    for (int i = l1-1; i >= 0; i--) {
      for (int j = l2-1; j >= 0; j--) {
	sub_pairProb = (s1[i] == s2[j]) ? m_matchLogProb : m_nonMatchLogProb;
	matrix[i][j][1] = logSum(m_gapTokenLogProb + m_gapExtendLogProb + matrix[i+1][j][1],
				 sub_pairProb + m_gapEndLogProb + matrix[i+1][j+1][0]);
	matrix[i][j][2] = logSum(m_gapTokenLogProb + m_gapExtendLogProb + matrix[i][j+1][2],
				 sub_pairProb + m_gapEndLogProb + matrix[i+1][j+1][0]);
	tmpLog = logSum(m_gapTokenLogProb + matrix[i+1][j][1], m_gapTokenLogProb + matrix[i][j+1][2]);
	matrix[i][j][0] = logSum(sub_pairProb + m_subLogProb + matrix[i+1][j+1][0],
				 m_gapStartLogProb + tmpLog);
      }
    }
    return matrix;
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
   * print out the three matrices
   */
  public void printMatrices(TokenString ts1, TokenString ts2) {
    double[][][] forward = forward(ts1, ts2);
    double[][][] backward = backward(ts1, ts2);
    int l1 = ts1.tokenIDs.length, l2 = ts2.tokenIDs.length;

    double totalForward = logSum(m_endAtSubLogProb + forward[l1][l2][0], m_endAtGapLogProb + forward[l1][l2][1]);
    totalForward = logSum(totalForward, m_endAtGapLogProb + forward[l1][l2][2]);
    System.out.println("\nB:" + backward[0][0][0] + "\tF:" + totalForward);
    
    System.out.println("\n***FORWARD***\nSUBSTITUTION:");
    printAlignmentMatrix(ts1, ts2, 0, forward);

    System.out.println("\n\nDELETION:");
    printAlignmentMatrix(ts1, ts2, 1, forward);

    System.out.println("\n\nINSERTION:");
    printAlignmentMatrix(ts1, ts2, 2, forward);


    System.out.println("\n***BACKWARD***\nSUBSTITUTION:");
    printAlignmentMatrix(ts1, ts2, 0, backward);

    System.out.println("\n\nDELETION:");
    printAlignmentMatrix(ts1, ts2, 1, backward);

    System.out.println("\n\nINSERTION:");
    printAlignmentMatrix(ts1, ts2, 2, backward);
  }

  public void printAlignmentMatrix(TokenString ts1, TokenString ts2, int idx, double[][][] matrix) {
    DecimalFormat fmt = new DecimalFormat ("0.0000");
    
    System.out.print('\t');
    for (int i = 0; i < ts2.tokenIDs.length; i++) {
      System.out.print("\t" + ts2.tokenIDs[i]);
    }
    System.out.println();
    for (int i = 0; i < matrix.length; i++) {
      if (i > 0) System.out.print(ts1.tokenIDs[i-1] + "\t");  else System.out.print("\t");
      for (int j = 0; j < matrix[i].length; j++) {
	System.out.print(fmt.format(matrix[i][j][idx]) + "\t");
      }
      System.out.println();
    }
  } 
  
        
  /**
   *  Train the distance parameters using provided examples using EM
   * @param matched_pairs Each member is a String[] extendaining two matching fields
   * @param matched_pairs Each member is a String[] extendaining two non-matching fields
   */ 
  public void trainMetric (ArrayList pairList) throws Exception {
    initProbs();
    recordCosts(0);

    // initialize m_stringTokenStringMap
    m_stringTokenStringMap = new HashMap();
    for (int j = 0; j < pairList.size(); j++) {
      StringPair pair = (StringPair)pairList.get(j);
      if (!m_stringTokenStringMap.containsKey(pair.str1)) { 
	m_stringTokenStringMap.put(pair.str1, m_tokenizer.getTokenString(pair.str1));
      }
      if (!m_stringTokenStringMap.containsKey(pair.str2)) { 
	m_stringTokenStringMap.put(pair.str2, m_tokenizer.getTokenString(pair.str2));
      }
//        System.out.println("Pair:\t" + pair.str1 + "\n\t\"" + m_stringTokenStringMap.get(pair.str1) + "\"\n\t" +
//    			 pair.str2 + "\n\t\"" + m_stringTokenStringMap.get(pair.str2) + "\"");
    }

    // initialize the token emission probabilities
    int numTokens = m_stringTokenStringMap.size();
    m_gapTokenProb = 1.0/numTokens;
    normalizeEmissionProbs();
    updateLogProbs();

    try {
      // dump out the current probablities
      PrintWriter out = new PrintWriter(new FileWriter("/tmp/probs1"));
      double totalProb = 0;
      double prevTotalProb = Double.MIN_VALUE;
      for (int i = 1; i <= m_numIterations && Math.abs(totalProb - prevTotalProb) > 1; i++) {
	resetOccurrences();
	out.println(i + ":\t" + m_endAtSubProb + "\t" + m_subProb + "\t" + m_gapStartProb +
		    "\t" + m_endAtGapProb + "\t" + m_gapEndProb + "\t" + m_gapExtendProb + "\t" + m_matchProb);

	// go through positives
	prevTotalProb = totalProb;
	totalProb = 0;
	int shortestIdx = 0; int shortest = 100; 
	for (int j = 0; j < pairList.size(); j++) {
	  StringPair pair = (StringPair)pairList.get(j);
	  if (pair.positive) {
//  	    System.out.println("Pair:\t" + pair.str1 + "\n\t\"" + m_stringTokenStringMap.get(pair.str1) + "\"\n\t" +
//  			 pair.str2 + "\n\t\"" + m_stringTokenStringMap.get(pair.str2) + "\"");
	    totalProb += expectationStep ((TokenString) m_stringTokenStringMap.get(pair.str1),
					  (TokenString) m_stringTokenStringMap.get(pair.str2), 1, true);
	    if (((TokenString) m_stringTokenStringMap.get(pair.str2)).tokenIDs.length < shortest) {
	      shortest = ((TokenString) m_stringTokenStringMap.get(pair.str2)).tokenIDs.length;
	      shortestIdx = j;
	    } 
	  }
	}
	// go through negatives  - TODO - discriminative training
	//	    for (int j = 0; j < negExamples.length; j++)
	//		expectationStep (negExamples[j][1], negExamples[j][0], 1, false);

	if (m_verbose) {
	  DecimalFormat fmt = new DecimalFormat ("0.000");
	  System.out.println("\n" + i + ". Total likelihood=" + fmt.format(totalProb) + ";  prev=" + fmt.format(prevTotalProb));
	  System.out.println("************ Accumulated expectations ******************** ");
	  System.out.println("End_s=" + fmt.format(m_endAtSubOccs) + "\tSub=" + fmt.format(m_subOccs) +
			     "\tStGap=" + fmt.format(m_gapStartOccs) + "\nEnd_g=" + fmt.format(m_endAtGapOccs) +
			     "\tEndGap=" + fmt.format(m_gapEndOccs) + " ContGap=" + fmt.format(m_gapExtendOccs) +
			     "\nmatch=" + fmt.format(m_matchOccs) + "\tnonMatch=" + fmt.format(m_nonMatchOccs));
	  System.out.println("********************************");
	}
	maximizationStep ();
//  	StringPair pair = (StringPair)pairList.get(shortestIdx);
//  	printMatrices((TokenString) m_stringTokenStringMap.get(pair.str1),(TokenString) m_stringTokenStringMap.get(pair.str1));
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
  protected double expectationStep (TokenString ts1, TokenString ts2, int lambda, boolean pos_training) {
    int [] s1 = ts1.tokenIDs;
    int [] s2 = ts2.tokenIDs;
    int l1 = s1.length, l2 = s2.length;
    if (l1 == 0 || l2 == 0) {
      return 0;
    }
    double fMatrix[][][] = forward (ts1, ts2);
    double bMatrix[][][] = backward (ts1, ts2);
    double stringProb = bMatrix[0][0][0];
//NB: b[0][0][0]must be equal to endAtSub*f[l1][l2][0] + endAtGap*(f[l1][l2][1]+f[l1][l2][2]); uncomment below for sanity check
//      double totalForward = logSum(m_endAtSubLogProb + fMatrix[l1][l2][0], m_endAtSubLogProb + fMatrix[l1][l2][1]);
//      totalForward = logSum(totalForward, m_endAtSubLogProb + fMatrix[l1][l2][2]);
//      System.out.println("b:" + bMatrix[0][0][0] + "\tf:" + totalForward);    

    double occsSubst, occsStartGap_1, occsStartGap_2, occsExtendGap_1, occsExtendGap_2;
    double occsEndGap_1, occsEndGap_2;
    double subTokenLogProb;
    int s1_i, s2_j;

    if (stringProb == 0.0) {
      System.out.println("TROUBLE!!!!  s1=" + ts1 + " s2=" + ts2);
      printMatrices(ts1,ts2);
      return 0;
    }
    m_endAtSubOccs += lambda; 
    m_endAtGapOccs += 2*lambda;

    for (int i = 1; i < l1; i++) {
      for (int j = 1; j < l2; j++) {
	s1_i = s1[i-1];
	s2_j = s2[j-1];
	if (s1_i == s2_j) {
	  subTokenLogProb = m_matchLogProb;
	} else {
	  subTokenLogProb = m_nonMatchLogProb;
	}

	// substituting or matching
	occsSubst = Math.exp(fMatrix[i-1][j-1][0] + subTokenLogProb + m_subLogProb + bMatrix[i][j][0] - stringProb);
	if (s1_i == s2_j) {
	  m_matchOccs += occsSubst;
	} else {
	  m_nonMatchOccs += occsSubst;
	}
	m_subOccs += occsSubst;
		
	// starting a gap
	occsStartGap_1 = Math.exp(fMatrix[i-1][j][0] + m_gapTokenLogProb + m_gapStartLogProb + bMatrix[i][j][1] - stringProb);
	occsStartGap_2 = Math.exp(fMatrix[i][j-1][0] + m_gapTokenLogProb + m_gapStartLogProb + bMatrix[i][j][2] - stringProb);
	m_gapStartOccs += occsStartGap_1 + occsStartGap_2;
		
	// extendinuing a gap     
	occsExtendGap_1 = Math.exp(fMatrix[i-1][j][1] + m_gapTokenLogProb + m_gapExtendLogProb + bMatrix[i][j][1] - stringProb);
	occsExtendGap_2 = Math.exp(fMatrix[i][j-1][2] + m_gapTokenLogProb + m_gapExtendLogProb + bMatrix[i][j][2] - stringProb);
	m_gapExtendOccs += occsExtendGap_1 + occsExtendGap_2;
		
	// ending a gap
	occsEndGap_1 = Math.exp(fMatrix[i-1][j-1][1] + subTokenLogProb + m_gapEndLogProb + bMatrix[i][j][0] - stringProb);
	if (s1_i == s2_j) {
	  m_matchOccs += occsEndGap_1;   // TODO - check!!!  , also if's above and below
	} else {
	  m_nonMatchOccs += occsEndGap_1;
	}
	occsEndGap_2 = Math.exp(fMatrix[i-1][j-1][2] + subTokenLogProb + m_gapEndLogProb + bMatrix[i][j][0] - stringProb);
	if (s1_i == s2_j) {
	  m_matchOccs += occsEndGap_2;
	} else {
	  m_nonMatchOccs += occsEndGap_2;
	}
	m_gapEndOccs += occsEndGap_1 + occsEndGap_2;
      }
    }
    // border rows.  We can't end gap, and can start/extend gap only one way
    for (int i = 1; i < l1; i++) {
      s1_i = s1[i-1];
      s2_j = s2[l2-1];
      if (s1_i == s2_j) {
	subTokenLogProb = m_matchLogProb;
      } else {
	subTokenLogProb = m_nonMatchLogProb;
      }

      occsSubst = Math.exp(fMatrix[i-1][l2-1][0] + subTokenLogProb + m_subLogProb + bMatrix[i][l2][0] - stringProb);
      if (s1_i == s2_j) {
	m_matchOccs += occsSubst;
      } else {
	m_nonMatchOccs += occsSubst;
      }
      m_subOccs += occsSubst;
		
      occsStartGap_1 = Math.exp(fMatrix[i-1][l2][0] + m_gapTokenLogProb + m_gapStartLogProb + bMatrix[i][l2][1] - stringProb);
      m_gapStartOccs += occsStartGap_1;

      occsExtendGap_1 = Math.exp(fMatrix[i-1][l2][1] + m_gapTokenLogProb + m_gapExtendLogProb + bMatrix[i][l2][1] - stringProb);
      m_gapExtendOccs += occsExtendGap_1;   //  DO WE NEED THIS??? WE HAD NO CHOICE!
    }
    for (int j = 1; j < l2; j++) {
      s1_i = s1[l1-1];
      s2_j = s2[j-1];
      if (s1_i == s2_j) {
	subTokenLogProb = m_matchLogProb;
      } else {
	subTokenLogProb = m_nonMatchLogProb;
      }

      occsSubst = Math.exp(fMatrix[l1-1][j-1][0] + subTokenLogProb + m_subLogProb + bMatrix[l1][j][0] - stringProb);
      if (s1_i == s2_j) {
	m_matchOccs += occsSubst;
      } else {
	m_nonMatchOccs += occsSubst;
      } 
      m_subOccs += occsSubst;
	    
      occsStartGap_2 = Math.exp(fMatrix[l1][j-1][0] + m_gapTokenLogProb + m_gapStartLogProb + bMatrix[l1][j][2] - stringProb);
      m_gapStartOccs += occsStartGap_2;
	    
      occsExtendGap_2 = Math.exp(fMatrix[l1][j-1][2] + m_gapTokenLogProb + m_gapExtendLogProb + bMatrix[l1][j][2] - stringProb);
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

    // Sum up expectations for occurrences in deletion/insertion states
    N = m_gapExtendOccs + m_gapEndOccs + m_endAtGapOccs;
    m_gapExtendProb = m_gapExtendOccs / N;
    m_gapEndProb = m_gapEndOccs / N;
    m_endAtGapProb = m_endAtGapOccs / N;

    // regularize if necessary
    if (m_subProb < m_clampProb) m_subProb = m_clampProb;
    if (m_gapStartProb < m_clampProb) m_gapStartProb = m_clampProb;
    if (m_endAtSubProb < m_clampProb) m_endAtSubProb = m_clampProb;
    if (m_gapExtendProb < m_clampProb) m_gapExtendProb = m_clampProb;
    if (m_gapEndProb < m_clampProb) m_gapEndProb = m_clampProb;
    if (m_endAtGapProb < m_clampProb) m_endAtGapProb = m_clampProb;
    if (m_endAtGapProb < m_clampProb) m_endAtGapProb = m_clampProb;

    m_matchProb = m_matchOccs / (m_matchOccs + m_nonMatchOccs);
    if (m_matchProb < m_clampProb) m_matchProb = m_clampProb;
    if (1.0 - m_matchProb < m_clampProb) m_matchProb = 1.0 - m_clampProb; 

    normalizeTransitionProbs();
    normalizeEmissionProbs();
    updateLogProbs();
  }
    

  /** 
   * Normalize the probabilities of emission editops so that they sum to 1
   * for each state */
  protected void normalizeEmissionProbs() {
    int numTokens = m_stringTokenStringMap.size();
    m_nonMatchProb = (1.0 - m_matchProb) / (numTokens * numTokens - numTokens);
  }


  
  /** 
   * Normalize the probabilities of transitions so that they sum to 1
   * for each state*/
  protected void normalizeTransitionProbs() {
    // M-state
    double P = m_subProb + 2 * m_gapStartProb + m_endAtSubProb;
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
    m_matchOccs = 0;
    m_nonMatchOccs = 0;
    m_endAtSubOccs = 0;
    m_endAtGapOccs = 0;
    m_gapStartOccs = 0;
    m_gapExtendOccs = 0;
    m_gapEndOccs = 0;
    m_subOccs = 0;
  }

  /** 
   *  initialize the probabilities to some startup values
   */
  protected void initProbs () {
    m_endAtSubProb = 0.05;
    m_endAtGapProb = 0.1;
    m_gapStartProb = 0.05;
    m_gapExtendProb = 0.5;
    m_gapEndProb = 0.4;
    m_subProb = 0.85;
    m_matchProb = 0.9;
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
    m_matchCost =     -m_matchLogProb;
    m_nonMatchCost =  -m_nonMatchLogProb;

    if (m_verbose) {
      System.out.println("\nScaled by extend cost:\nGapStrt=" + (m_gapStartCost/m_gapExtendCost) +
			 "\tGapExt=" + (m_gapExtendCost/m_gapExtendCost) +
			 "\tGapEnd=" + (m_gapEndCost/m_gapExtendCost) + 
			 "\tSub=" + (m_subCost/m_gapExtendCost) +
			 "\tNoop=" + (m_matchCost/m_gapExtendCost));
      System.out.println("\nActual costs:\nGapStrt=" + (m_gapStartCost) +
			 "\tGapExt=" + (m_gapExtendCost) +
			 "\tGapEnd=" + (m_gapEndCost) + 
			 "\tSub=" + (m_subCost) +
			 "\tNoop=" + (m_matchCost));

    }
  }

  /**
   * store logs of all probabilities in m_editopLogProbs
   */
  protected void updateLogProbs() {
    m_matchLogProb = Math.log(m_matchProb);
    m_nonMatchLogProb = Math.log(m_nonMatchProb);
    m_gapTokenLogProb = Math.log(m_gapTokenProb);
    m_endAtSubLogProb = Math.log(m_endAtSubProb);
    m_endAtGapLogProb = Math.log(m_endAtGapProb);
    m_gapStartLogProb = Math.log(m_gapStartProb);
    m_gapExtendLogProb = Math.log(m_gapExtendProb);
    m_gapEndLogProb = Math.log(m_gapEndProb);
    m_subLogProb = Math.log(m_subProb);

    DecimalFormat fmt = new DecimalFormat ("0.0000");
    if (m_verbose) { 
      System.out.println("After update:\tNOOP=" + fmt.format(m_matchProb) + "=" + fmt.format(m_matchLogProb) +
			 "\tSUB=" + fmt.format(m_subProb) + "=" + fmt.format(m_subLogProb) +
			 "\n\t\tGAPst=" + fmt.format(m_gapStartProb) + "=" + fmt.format(m_gapStartLogProb) +
			 "\tGAPcont=" + fmt.format(m_gapExtendProb) + "=" + fmt.format(m_gapExtendLogProb) +
			 "\tGAPend=" + fmt.format(m_gapEndProb) + "=" + fmt.format(m_gapEndLogProb) +
			 "\n\t\tendAtGap=" + fmt.format(m_endAtGapProb) + "=" + fmt.format(m_endAtGapLogProb) +
			 "\tendAtSub=" + fmt.format(m_endAtSubProb) + "=" + fmt.format(m_endAtSubLogProb));
    }
  }

    
  /** 
   * Get the distance between two strings
   * @param s1 first string
   * @param s2 second string
   * @return a value of this distance between these two strings
   */
  public double distance (String s1, String s2) {
    if (m_useGenerativeModel) {

      // retrieve the tokenstring's
      TokenString ts1;
      if (m_stringTokenStringMap.containsKey(s1)) {
	ts1 = ((TokenString)m_stringTokenStringMap.get(s1));
      } else {
	ts1 = m_tokenizer.getTokenString(s1);
	m_stringTokenStringMap.put(s1, ts1);
      }
      
      TokenString ts2;
      if (m_stringTokenStringMap.containsKey(s2)) {
	ts2 = ((TokenString)m_stringTokenStringMap.get(s2));
      } else {
	ts2 = m_tokenizer.getTokenString(s2);
	m_stringTokenStringMap.put(s2, ts2);
      }
    
      double d = backward(ts1,ts2)[0][0][0];
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
    TokenString ts1;
    if (m_stringTokenStringMap.containsKey(string1)) {
      ts1 = ((TokenString)m_stringTokenStringMap.get(string1));
    } else {
      ts1 = m_tokenizer.getTokenString(string1);
      m_stringTokenStringMap.put(string1, ts1);
    }

    TokenString ts2;
    if (m_stringTokenStringMap.containsKey(string2)) {
      ts2 = ((TokenString)m_stringTokenStringMap.get(string2));
    } else {
      ts2 = m_tokenizer.getTokenString(string2);
      m_stringTokenStringMap.put(string2, ts2);
    }
    
    int [] s1 = ts1.tokenIDs; 
    int [] s2 = ts2.tokenIDs;
    int l1 = s1.length, l2 = s2.length;
    double T[][] = new double[l1+1][l2+1];
    double I[][] = new double[l1+1][l2+1];
    double D[][] = new double[l1+1][l2+1];
    double subCost = 0, subTokenCost = 0, ret;
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
      T[0][j] = T[0][j-1] + m_gapExtendCost;
    }
    for (j = 2; j < l1+1; j++) {
      T[j][0] = T[j-1][0] + m_gapExtendCost;
    }
    for (i = 1; i < l1+1; i++) {
      for (j = 1; j < l2+1; j++) {
	int t1 = s1[i-1];
	int t2 = s2[j-1];
	subTokenCost = (t1 == t2) ? m_matchCost : m_nonMatchCost;  // TODO:  experiment with 0 matchCost

	if (D[i-1][j]+m_gapExtendCost > T[i-1][j]+m_gapStartCost) {
	  D[i][j] = T[i-1][j]+m_gapStartCost;
	} else {
	  D[i][j] = D[i-1][j]+m_gapExtendCost;
	}
		
	if (I[i][j-1]+m_gapExtendCost > T[i][j-1]+m_gapStartCost) {
	  I[i][j] = T[i][j-1] + m_gapStartCost;
	} else {
	  I[i][j] = I[i][j-1] + m_gapExtendCost;
	}
		
	//	subCost = m_subCost + sub_charCost;
//  	subCost =((c1 == c2) ? 0 : (m_subCost + m_gapEndCost));
//  	subCost = subCost + sub_charCost;
//    	subCost = (c1 == c2) ? 0 : m_subCost;
	//	subCost = m_subCost;
		
	if  ((T[i-1][j-1] + m_subCost < D[i-1][j-1] + m_gapEndCost) &&    /// d[i][j] or d[i-1][j-1]??
	     (T[i-1][j-1] + m_subCost < I[i-1][j-1] + m_gapEndCost )) {
	  T[i][j] = T[i-1][j-1] + m_subCost + subTokenCost;   // ?? do we add subCharCost?
	} else {
	  if (D[i-1][j-1] < I[i-1][j-1]) {
	    T[i][j] = D[i-1][j-1] + m_gapEndCost + subTokenCost;
	  } else {
	    T[i][j] = I[i-1][j-1] + m_gapEndCost + subTokenCost;
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
    LearnableTokenEDAffine metric = new LearnableTokenEDAffine();
    metric.setNormalized(m_normalized);
    metric.setTokenizer(m_tokenizer);
    System.out.println("Tokenizer: + "+ ((WordTokenizer)m_tokenizer).getStopwordRemoval());
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
    String [] options = new String [40];
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

    options[current++] = "-T";
    options[current++] = Utils.removeSubstring(m_tokenizer.getClass().getName(), "weka.deduping.metrics.");
    if (m_tokenizer instanceof OptionHandler) {
      String[] tokenizerOptions = ((OptionHandler)m_tokenizer).getOptions();
      for (int i = 0; i < tokenizerOptions.length; i++) {
	options[current++] = tokenizerOptions[i];
      }
    }

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

    System.out.println("Setting options - BZZZZ!");
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

  /** Set the tokenizer to use
   * @param tokenizer the tokenizer that is used
   */
  public void setTokenizer(Tokenizer tokenizer) {
    m_tokenizer = tokenizer;
  }

  /** Get the tokenizer to use
   * @return the tokenizer that is used
   */
  public Tokenizer getTokenizer() {
    return m_tokenizer;
  }



    
  public static void main(String[] args) {
    try { 
    LearnableTokenEDAffine metric = new LearnableTokenEDAffine();
    //    metric.trainMetric(new ArrayList());
    Tokenizer tokenizer = new WordTokenizer();
    TokenString ts1 = tokenizer.getTokenString("Matthew Turk and Alex");
    TokenString ts2 = tokenizer.getTokenString("Matthew Turk and Alex");
    metric.printMatrices(ts1, ts2);
    } catch (Exception e) { e.printStackTrace();}
  }
    
}
