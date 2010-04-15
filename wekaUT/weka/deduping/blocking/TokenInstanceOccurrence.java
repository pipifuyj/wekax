package weka.deduping.blocking;

/** A lightweight object for storing information about an occurrence of a token (a.k.a word, term)
 * in a record.
 *
 * @author Ray Mooney
 */

public class TokenInstanceOccurrence {
    /** A reference to the Document where it occurs */
    public InstanceReference instanceRef = null;
    /** The number of times it occurs in the Document */
    public int count = 0;

    /** Create an occurrence with these values */
    public TokenInstanceOccurrence(InstanceReference instanceRef, int count) {
	this.instanceRef = instanceRef;
	this.count = count;
    }
}
