package weka.deduping.metrics;

/** A lightweight object for storing information about an occurrence of a token (a.k.a word, term)
 * in a Document.
 *
 * @author Ray Mooney
 */

public class TokenOccurrence {
    /** A reference to the Document where it occurs */
    public StringReference m_stringRef = null;
    /** The number of times it occurs in the Document */
    public int m_count = 0;

    /** Create an occurrence with these values */
    public TokenOccurrence(StringReference stringRef, int count) {
	m_stringRef = stringRef;
	m_count = count;
    }
}
