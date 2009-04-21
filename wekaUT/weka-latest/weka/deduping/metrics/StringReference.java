package weka.deduping.metrics;

import java.io.*;

/** A simple data structure for storing a reference to a document file
 *  that includes information on the length of its document vector.
 *  The goal is to have a lightweight object to store in an inverted index
 *  without having to store an entire Document object.
 *
 * @author Ray Mooney
 */

public class StringReference {
  /** The referenced string. */
  public String m_string = null;
  /** The corresponding HashMapVector */
  public HashMapVector m_vector = null;
  /** The length of the corresponding Document vector. */
  public double m_length = 0.0;

  

  public StringReference(String string, HashMapVector vector, double length) {
    m_string = string;
    m_vector = vector;
    m_length = length;
  }

  /** Create a reference to this document, initializing its length to 0 */
  public StringReference(String string, HashMapVector vector) {
    this(string, vector, 0.0);
  }

  public String toString() {
    return m_string;
  }
}
