package weka.datagenerators;

import java.util.Collection;

/**
 * The interface that all filters have to implement.  A token filter
 * transforms a given token into another form.  It may also toss
 * unused tokens out.
 *
 * @author ywwong
 * @version $Id: TokenFilter.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
interface TokenFilter {

    /**
     * Applies the token filter.
     *
     * @param strToken  The input token
     * @return The transformed token; or <code>null</code> if the
     * input token is being tossed out.
     */
    public String apply(String strToken);

    ////// WEKA specific. //////

    public Collection getOptions();

}
