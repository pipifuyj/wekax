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
 *    Tokenizer.java
 *    Copyright (C) 2001 Mikhail Bilenko
 *
 */


package weka.deduping.metrics;

import java.util.*;
import java.io.*;

/**
 * This abstract class defines a tokenizer that turns strings into HashMapVectors
 *
 * @author Mikhail Bilenko
 */
public abstract class Tokenizer {

  /** A tokenizer keeps an index of seen tokens */ 
  protected HashMap m_stringIDmap;
  protected int m_currIDidx;

  public Tokenizer() {
    m_stringIDmap = new HashMap();
  }
  
    
  /** Take a string and create a vector of tokens from it
   * @param string a String to tokenize
   * @return vector with individual tokens
   */
  public abstract HashMapVector tokenize(String s);


  /** Take a string, a create a TokenString out of it
   * @param string a string to tokenize
   * @return a TokenString
   */
  public abstract TokenString getTokenString(String s); 


}







