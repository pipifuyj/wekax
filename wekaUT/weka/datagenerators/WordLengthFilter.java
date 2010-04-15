package weka.datagenerators;

import weka.core.Option;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Tosses tokens that are outside a certain range of length.
 *
 * <p><b>WEKA options:</b>
 * <ul>
 *   <li><code>-N &lt;num&gt;</code> - Specifies the minimum length.
 *   This parameter has no default value.
 *
 *   <li><code>-X &lt;num&gt;</code> - Specifies the maximum length.
 *   This parameter has no default value.
 * </ul>
 *
 * <p>It is a fatal error to leave both of the parameters unspecified.
 *
 * @author ywwong
 * @version $Id: WordLengthFilter.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
class WordLengthFilter implements TokenFilter {

    /** Words that are shorter than this will be ignored. */
    protected int m_nMin;

    /** Words that are longer than this will be ignored. */
    protected int m_nMax;

    ////// WEKA specific. //////

    /** The option string for minimum length. */
    protected String m_strMin;

    /** The option string for maximum length. */
    protected String m_strMax;

    ////// Ends WEKA specific. //////

    /**
     * Creates a word length filter.
     *
     * @param ts  The TextSource object.
     */
    public WordLengthFilter(TextSource ts, String[] options) throws Exception {
        int nMin;
        int nMax;
        Integer n;

        ////// WEKA specific. //////

        m_strMin = Utils.getOption('N', options);
        if (m_strMin.length() == 0)
            nMin = -1;
        else {
            n = Integer.valueOf(m_strMin);
            if (n == null || n.intValue() < 0)
                throw new Exception("Invalid minimum width (-N).");
            else
                nMin = n.intValue();
        }
        m_strMax = Utils.getOption('X', options);
        if (m_strMax.length() == 0)
            nMax = -1;
        else {
            n = Integer.valueOf(m_strMax);
            if (n == null || n.intValue() < 0)
                throw new Exception("Invalid maximum width (-X).");
            else
                nMax = n.intValue();
        }

        ////// Ends WEKA specific. //////

        if (nMin < 0 && nMax < 0)
            throw new Exception("Neither widths are set (-X or -N).");
        if (nMin >= 0 && nMax >= 0 && nMin > nMax)
            throw new Exception("Invalid range (-X and -N).");
        m_nMin = nMin;
        m_nMax = nMax;
    }

    /** Tosses tokens that are shorter than the minimum length.
     *
     * @param strToken  The input token
     * @return The input token; <code>null</code> if the length of the
     * input token is out of range
     */
    public String apply(String strToken) {
        int nLen;

        nLen = strToken.length();
        if ((m_nMin >= 0 && nLen < m_nMin) || (m_nMax >= 0 && nLen > m_nMax))
            return null;
        else
            return strToken;
    }

    ////// WEKA specific. //////

    public static Collection listOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        aOpts.add(new Option("\tWordLengthFilter: Minimum length " +
                             "(default unbounded)",
                             "N", 1, "-N <num>"));
        aOpts.add(new Option("\tWordLengthFilter: Maximum length " +
                             "(default unbounded)",
                             "X", 1, "-X <num>"));
        return aOpts;
    }

    public Collection getOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        if (m_strMin.length() > 0) {
            aOpts.add("-N");
            aOpts.add(m_strMin);
        }
        if (m_strMax.length() > 0) {
            aOpts.add("-X");
            aOpts.add(m_strMax);
        }
        return aOpts;
    }

}
