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
 *    TokenString.java
 *    This class wraps a string of tokens
 *    Copyright (C) 2004 Mikhail Bilenko
 *
 */


package weka.deduping.metrics;

import java.util.*;
import java.io.Serializable;
import weka.core.*;


public class TokenString {
  protected String m_string = null;
  public String[] tokens = null;
  public int[] tokenIDs = null;

  public TokenString(String s) {
    m_string = s; 
  }

  public String toString() {
    StringBuffer s = new StringBuffer();
    if (tokenIDs != null) { 
      for (int i = 0; i < tokenIDs.length-1; i++) {
	s.append(tokenIDs[i] + "(" + tokens[i] + ").");
      }
      s.append(tokenIDs[tokenIDs.length-1]);
    }
    return s.toString();
  } 
} 



