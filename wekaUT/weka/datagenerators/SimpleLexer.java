package weka.datagenerators;

import weka.core.Option;
import weka.core.Utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Splits a given input string by matching a particular pattern.
 * There are three lexing styles, namely <i>whitespace</i>,
 * <i>alphanum</i>, and <i>alpha</i>.
 *
 * <p>The <i>whitespace</i> style keeps all strings that begin and end
 * with non-whitespace characters, separated by whitespace.  For
 * example, the string "Austin, TX 78712-1188" will result in the
 * tokens "Austin,", "TX," and "78712-1188".
 *
 * <p>The <i>alphanum</i> style keeps all alphanumeric strings, separated
 * by non-alphanumeric characters.  For example, the string "Austin,
 * TX 78712-1188" will result in the tokens "Austin", "TX", "78712"
 * and "1188".
 *
 * <p>The <i>alpha</i> style keeps all alphabetic strings, separated by
 * non-alphabetic characters.  For example, the string "Austin, TX
 * 78712-1188" will result in the tokens "Austin" and "TX".
 *
 * <p><b>WEKA options:</b>
 * <ul>
 *   <li><code>-y &lt;str&gt;</code> - The lexing style, which is
 *   one of <code>whitespace</code>, <code>alphanum</code>, or
 *   <code>alpha</code>.  This parameter has no default value and is
 *   not optional.
 * </ul>
 *
 * @author ywwong
 * @version $Id: SimpleLexer.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
class SimpleLexer extends Lexer {

    public static final int WHITESPACE = 0;
    public static final int ALPHANUM = 1;
    public static final int ALPHA = 2;

    /** Unit of size whereby the character buffer is increased. */
    protected static final int INC = 50;

    /** The character buffer. */
    protected char[] m_buf;

    /** The lexing style. */
    protected int m_nStyle;

    ////// WEKA specific. //////

    protected String m_strStyle;

    ////// Ends WEKA specific. //////

    /**
     * Creates a simple lexer.
     *
     * @param ts      The TextSource object.
     * @param reader  The document reader.
     */
    public SimpleLexer(TextSource ts, DocumentReader reader,
                       String[] options) throws Exception {
        super(reader);

        ////// WEKA specific. //////

        m_strStyle = Utils.getOption('y', options);
        if (m_strStyle.length() == 0)
            throw new Exception("Style (-y) not set.");
        else if (m_strStyle.equals("whitespace"))
            m_nStyle = WHITESPACE;
        else if (m_strStyle.equals("alphanum"))
            m_nStyle = ALPHANUM;
        else if (m_strStyle.equals("alpha"))
            m_nStyle = ALPHA;
        else
            throw new Exception("Invalid style (-y): \'" + m_strStyle + "\'.");

        ////// Ends WEKA specific. //////

        m_buf = new char[INC];
    }

    /**
     * Parses the next token from the input string.
     * 
     * @return The next token if it's available; <code>null</code> if
     * otherwise.
     */
    public String nextToken() throws IOException {
        char ch = 0;
        boolean b;
        int c;
        int i;

        // Skip separator.
        c = m_reader.read();
        while (c >= 0) {
            b = false;
            ch = (char) c;
            switch (m_nStyle) {
            case WHITESPACE:
                b = !Character.isWhitespace(ch);
                break;
            case ALPHANUM:
                b = Character.isLetterOrDigit(ch);
                break;
            case ALPHA:
                b = Character.isLetter(ch);
                break;
            }
            if (b)
                break;
            c = m_reader.read();
        }

        if (c < 0)
            return null;

        // Find the token.
        i = 1;
        m_buf[0] = ch;
        c = m_reader.read();
        while (c >= 0) {
            b = false;
            ch = (char) c;
            switch (m_nStyle) {
            case WHITESPACE:
                b = Character.isWhitespace(ch);
                break;
            case ALPHANUM:
                b = !Character.isLetterOrDigit(ch);
                break;
            case ALPHA:
                b = !Character.isLetter(ch);
                break;
            }
            if (b)
                break;

            if (i == m_buf.length) {
                char[] newBuf = new char[i + INC];

                for (int j = 0; j < i; j++)
                    newBuf[j] = m_buf[j];
                m_buf = newBuf;
            }
            m_buf[i++] = ch;
            c = m_reader.read();
        }
        return new String(m_buf, 0, i);
    }

    ////// WEKA specific. //////

    public static Collection listOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        aOpts.add(new Option("\tSimpleLexer: Lexing style",
                             "y", 1, "-y <str>"));
        return aOpts;
    }

    public Collection getOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        aOpts.add("-y");
        aOpts.add(m_strStyle);
        return aOpts;
    }

}
