package weka.datagenerators;

import weka.datagenerators.TextSource.Real;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * The interface that all document readers have to implement.
 * A document consists of a piece of text, and belongs to a particular
 * class.
 *
 * @author ywwong
 * @version $Id: DocumentReader.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
interface DocumentReader {

    /**
     * Tests if there are any unread documents.
     *
     * @return <code>true</code> if there are unread documents;
     * <code>false</code> if otherwise.
     */
    public boolean hasNextDocument();

    /**
     * Resets the reader so that when <code>read()</code> is called,
     * the next document is read.
     *
     * @return The class index of the next document; or
     * <code>null</code> if there are no more documents.
     */
    public Real nextDocument() throws IOException;

    /**
     * Reads the next character from the current document.
     *
     * @return The next character; or -1 if
     * <code>nextDocument()</code> has never been called, or there are
     * no more characters for the current document.
     */
    public int read() throws IOException;

    ////// WEKA specific. //////

    public Collection getOptions();

}
