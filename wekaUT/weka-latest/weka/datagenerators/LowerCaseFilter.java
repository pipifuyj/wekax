package weka.datagenerators;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Transforms a given token into lowercase.  It has no parameters.
 *
 * @author ywwong
 * @version $Id: LowerCaseFilter.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
class LowerCaseFilter implements TokenFilter {

    /**
     * Creates a lowercase filter.
     *
     * @param ts  The TextSource object.
     */
    public LowerCaseFilter(TextSource ts, String[] options) throws Exception
    { }

    /**
     * Transforms a given token into lowercase.
     *
     * @param strToken  The input token
     * @return The input token in lowercase.
     */
    public String apply(String strToken) {
        return strToken.toLowerCase();
    }

    ////// WEKA specific. //////

    public static Collection listOptions() {
        return new ArrayList();
    }

    public Collection getOptions() {
        return new ArrayList();
    }

}
