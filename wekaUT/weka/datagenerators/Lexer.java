package weka.datagenerators;

import java.io.IOException;
import java.util.Collection;

/**
 * Defines common behaviors for lexers.  A lexer splits a given input
 * string into an array of string tokens, according to some particular
 * rules.
 *
 * @author ywwong
 * @version $Id: Lexer.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
abstract class Lexer {

    /** The document reader. */
    protected DocumentReader m_reader;

    public Lexer(DocumentReader reader) { m_reader = reader; }

    /** Parses the next token from the document reader.
     * 
     * @return The next token if it's available; <code>null</code> if
     * otherwise.
     */
    public abstract String nextToken() throws IOException;

    ////// WEKA specific. //////

    public abstract Collection getOptions();

}
