package weka.datagenerators;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SparseInstance;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;
import java.util.regex.Pattern;

/** 
 * Reads a collection of text documents and transforms them into
 * sparse vectors.  The sparse vectors are then put into an ARFF file
 * for further processing by WEKA.
 *
 * <p><b>WEKA options:</b>
 * <ul>
 *   <li><code>-I</code> - Include TFIDF scores instead of TF.
 *
 *   <li><code>-R &lt;str&gt;</code> - The document reader.  Now only
 *   one is supported, namely <code>directory</code>.  This parameter
 *   has no default value and is not optional.
 *
 *   <li><code>-L &lt;str&gt;</code> - The lexer.  Now only one lexer
 *   is supported, namely <code>simple</code>.  This parameter has no
 *   default value and is not optional.
 *
 *   <li><code>-F &lt;str&gt;[:&lt;str&gt;...]</code> - A
 *   colon-separated list of filters being applied on the tokens.
 *   Four filters are supported, namely <code>lower_case</code>,
 *   <code>porter_stemmer</code>, <code>stop_word</code>, and
 *   <code>word_length</code>.  Order of listing is significant.  For
 *   example, if the value for <code>filters</code> is
 *   <code>stop_word:porter_stemmer</code>, then the
 *   <code>stop_word</code> filter is applied before
 *   <code>porter_stemmer</code>.  By default the list is empty.
 *
 *   <li>Document readers, filters and lexers have their own
 *   parameters.  See their documentation for detail.
 * </ul>
 *
 * <p>The generic generator options <code>-a</code>, <code>-c</code>
 * and <code>-n</code> are ignored.
 *
 * <p>Here are some sample command lines:
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y whitespace -o news.arff
 </pre>
 *
 * <p>The name of the dataset is <code>news</code>.  We use the
 * <code>directory</code> document reader.  The directory being read
 * is <code>cmu-newsgroup-random-100/</code>.  We use the
 * <code>simple</code> lexer and all tokens are delimited by
 * whitespace.  The output file is <code>news.arff</code>.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alphanum -o news.arff
 </pre>
 *
 * <p>In this case all tokens consist of only alphanumeric characters.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alpha -o news.arff
 </pre>
 *
 * <p>All tokens consist of only alphabets.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alpha -F lower_case -o news.arff
 </pre>
 *
 * <p>All tokens are converted to lower case before being indexed.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alpha -F lower_case:stop_word -o news.arff
 </pre>
 *
 * <p>All stop words are removed.  The default SMART stop list is used.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alpha -F lower_case:stop_word:porter_stemmer -o news.arff
 </pre>
 *
 * <p>After removing the stop words, we apply the Porter stemmer.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alpha
    -F lower_case:stop_word:porter_stemmer:word_length -N 5 -o news.arff
 </pre>
 *
 * <p>After stemming the tokens, we throw away all tokens whose length
 * is less than five.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/
    -L simple -y alpha
    -F lower_case:stop_word:word_length:porter_stemmer -N 5 -o news.arff
 </pre>
 *
 * <p>We throw away tokens whose length is less than five before
 * applying the Porter stemmer.
 *
 <pre>
  java weka.datagenerators.TextSource
    -r news -R directory -D cmu-newsgroup-random-100/ -u 'talk.*'
    -L simple -y alpha
    -F lower_case:stop_word:word_length:porter_stemmer -N 5 -o news.arff
 </pre>
 *
 * <p>Read only documents that belong to the classes
 * <code>talk.*</code>.  The argument for <code>-u</code> can be any
 * regular expression.
 *
 * @author ywwong
 * @version $Id: TextSource.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
public class TextSource extends Generator
    implements OptionHandler, Serializable {

    /** A simpler wrapper for int than Integer. */
    public class Int implements Comparable {
        public int m_i;
        public Int(int i) { m_i = i; }
        public int compareTo(Object o) {
            Int n = (Int) o;
            if (m_i < n.m_i)
                return -1;
            else if (m_i == n.m_i)
                return 0;
            else
                return 1;
        }
        public boolean equals(Object n) {
            if (n.getClass() == Int.class)
                return m_i == ((Int) n).m_i;
            else
                return false;
        }
        public int hashCode() { return m_i; }
        public String toString() { return Integer.toString(m_i); }
    }

    /** A simpler wrapper for double than Double. */
    public class Real {
        public double m_d;
        public Real(double d) { m_d = d; }
        public String toString() { return Double.toString(m_d); }
    }

    /** Sparse map data row structure with public hash map. */
    public class DataRow {
        public double m_dClass;
        public TreeMap m_data;
        public DataRow() {
            m_data = new TreeMap();
        }
        public void set(Int nIndex, Real dVal) {
            if (dVal.m_d == 0.0)
                m_data.remove(nIndex);
            else
                m_data.put(nIndex, dVal);
        }
        public void setClass(Real dClass) {
            m_dClass = dClass.m_d;
        }
        // WEKA specific.
        public Instance makeInstance(Table table) {
            Instance inst;
            double[] aVals;
            int[] aIndices;
            Iterator it;
            Entry ent;
            int i;

            aVals = new double[m_data.size() + 1];
            aIndices = new int[m_data.size() + 1];
            it = m_data.entrySet().iterator();
            for (i = 0; it.hasNext(); i++) {
                ent = (Entry) it.next();
                aVals[i] = ((Real) ent.getValue()).m_d;
                aIndices[i] = ((Int) ent.getKey()).m_i;
            }
            aVals[i] = m_dClass;
            aIndices[i] = table.m_nIndex;
            inst = new SparseInstance(0.0, aVals, aIndices, 
                                      table.m_format.numAttributes());
            inst.setDataset(table.m_format);
            return inst;
        }
    }

    /** Table that allows incremental addition of attributes. */
    public class Table {
        protected TextSource m_ts;
        protected FastVector m_attribs;
        protected Instances m_format;
        protected LinkedList m_data;
        protected int m_nIndex;       // class index
        protected ListIterator m_it;  // used by getNextInstance()
        public Table(TextSource ts) {
            m_ts = ts;
            m_attribs = new FastVector();
            m_data = new LinkedList();
            m_it = null;
        }
        public void add(DataRow vector) {
            m_data.add(vector);
        }
        public void addAttribute(Attribute attrib) {
            m_attribs.addElement(attrib);
        }
        // WEKA specific.
        public Instances makeDataFormat() throws Exception {
            FastVector attribs;
            FastVector aClasses;
            Set setKeys;
            Iterator it;

            // Add class index as one of the attributes.
            aClasses = new FastVector(m_ts.m_hashClasses.size());
            setKeys = m_ts.m_hashClasses.keySet();
            for (it = setKeys.iterator(); it.hasNext(); )
                aClasses.addElement(it.next());
            m_nIndex = m_attribs.size();
            attribs = (FastVector) m_attribs.copy();
            attribs.addElement(new Attribute("__class__", aClasses));
            m_format = new Instances(m_ts.getRelationName(), attribs, 0);
            m_format.setClassIndex(m_nIndex);

            // Update generator variables.
            m_ts.setNumClasses(aClasses.size());
            m_ts.setNumExamples(m_data.size());
            m_ts.setNumExamplesAct(m_data.size());
            m_ts.setNumAttributes(m_attribs.size() + 1);

            return m_format;
        }
        // WEKA specific.
        public Instance getNextInstance() {
            if (m_it == null)
                m_it = m_data.listIterator();
            return ((DataRow) m_it.next()).makeInstance(this);
        }
    }

    /** Information about a particular token. */
    protected class Token {
        /** The token string. */
        public String m_strToken;
        /** The token ID, which is the same as the attribute index. */
        public Int m_nID;
        /** The document frequency. */
        public int m_nDF;
        public Token(String strToken, Int nID) {
            m_strToken = strToken;
            m_nID = nID;
            m_nDF = 0;
        }
    }

    /** The example table. */
    protected Table m_table;

    /** A map for looking up tokens. */
    protected HashMap m_hashTokens;

    /** An ordered list for looking up tokens. */
    protected ArrayList m_aTokens;

    /** The next token ID. */
    protected int m_nNextToken;

    /** A map for looking up classes. */
    protected LinkedHashMap m_hashClasses;

    /** The next class ID. */
    protected double m_dNextClass;

    /** Collect TFIDF statistics instead of TF. */
    protected boolean m_bTFIDF;

    /** The document reader. */
    protected DocumentReader m_reader;

    /** The lexer. */
    protected Lexer m_lexer;

    /** The list of token filters which are applied in order. */
    protected LinkedList m_lstFilters;

    public TextSource() {
        m_table = new Table(this);
        m_hashTokens = new HashMap();
        m_aTokens = new ArrayList();
        m_nNextToken = 0;
        m_hashClasses = new LinkedHashMap();
        m_dNextClass = 0.0;
        m_bTFIDF = false;
        m_bFormatDefined = false;
    }

    // Called by document readers.
    public Real registerClass(String strClass) {
        Real dClass;

        dClass = (Real) m_hashClasses.get(strClass);
        if (dClass == null) {
            dClass = new Real(m_dNextClass);
            m_dNextClass += 1.0;
            m_hashClasses.put(strClass, dClass);
        }
        return dClass;
    }

    /**
     * Tokenizes a document and transforms it into a sparse vector.
     *
     * @param dClass  The class index of the document to be read.
     */
    protected DataRow getInstance(Real dClass) throws IOException {
        DataRow vector;
        String strToken;
        ListIterator itFilter;
        TokenFilter filter;

        Token token;
        Int nTokenID;
        Real dTF;
        Attribute attrib;

        Set setKeys;
        Iterator itKey;

        vector = new DataRow();
        vector.setClass(dClass);

        strToken = m_lexer.nextToken();
        while (strToken != null) {
            // Push token through the filters.
            for (itFilter = m_lstFilters.listIterator();
                 itFilter.hasNext(); ) {
                filter = (TokenFilter) itFilter.next();
                strToken = filter.apply(strToken);
                if (strToken == null)
                    break;
            }

            if (strToken != null) {
                // Update token info.
                token = (Token) m_hashTokens.get(strToken);
                if (token != null)
                    nTokenID = token.m_nID;
                else {
                    nTokenID = new Int(m_nNextToken++);
                    token = new Token(strToken, nTokenID);
                    attrib = new Attribute(strToken);
                    m_table.addAttribute(attrib);
                    m_hashTokens.put(strToken, token);
                    // The token with ID n can be found in m_aTokens[n].
                    m_aTokens.add(token);
                }

                // Update sparse vector.
                dTF = (Real) vector.m_data.get(nTokenID);
                if (dTF != null)
                    dTF.m_d += 1.0;
                else
                    vector.m_data.put(nTokenID, new Real(1.0));
            }
            strToken = m_lexer.nextToken();
        }

        // Update token info again.
        setKeys = vector.m_data.keySet();
        for (itKey = setKeys.iterator(); itKey.hasNext(); ) {
            nTokenID = (Int) itKey.next();
            token = (Token) m_aTokens.get(nTokenID.m_i);
            ++token.m_nDF;
        }
        return vector;
    }

    /**
     * Reads all documents and converts them all to sparse vectors.
     */
    protected void readInstances() throws Exception {
        DataRow vector;

        while (m_reader.hasNextDocument()) {
            vector = getInstance(m_reader.nextDocument());
            m_table.add(vector);
        }
        // Convert to TFIDF if necessary.
        if (m_bTFIDF) {
            Iterator itr, itw;
            DataRow row;
            Entry ent;
            Real r;
            Token t;
            double nDocs, max, d;
            int nTokens, i;

            nDocs = m_table.m_data.size();
            nTokens = m_aTokens.size();
            for (itr = m_table.m_data.iterator(); itr.hasNext(); ) {
                row = (DataRow) itr.next();
                max = 0.0;
                for (itw = row.m_data.entrySet().iterator(); itw.hasNext(); ) {
                    ent = (Entry) itw.next();
                    if (((Int) ent.getKey()).m_i < nTokens) {
                        d = ((Real) ent.getValue()).m_d;
                        if (max < d)
                            max = d;
                    }
                }
                for (itw = row.m_data.entrySet().iterator(); itw.hasNext(); ) {
                    ent = (Entry) itw.next();
                    i = ((Int) ent.getKey()).m_i;
                    if (i < nTokens) {
                        r = (Real) ent.getValue();
                        t = ((Token) m_aTokens.get(i));
                        r.m_d /= max;
                        r.m_d *= Math.log(nDocs / t.m_nDF);
                    }
                }
            }
        }
    }

    ////// WEKA specific stuff. //////

    /** The option string for document reader. */
    protected String m_strDocReader;

    /** The option string for lexer. */
    protected String m_strLexer;

    /** The option string for token filters. */
    protected String m_strFilters;

    /** True iff defineDataFormat() has been called. */
    protected boolean m_bFormatDefined;

    public String globalInfo() {
        return
            "A data generator that reads a collection of text documents " +
            "and transforms them into sparse vectors.";
    }

    public Enumeration listOptions() {
        Vector aOpts;

        aOpts = new Vector();
        aOpts.add(new Option("\tCompute TFIDF instead of TF (default false)",
                             "I", 0, "-I"));
        aOpts.add(new Option("\tDocument reader", "R", 1, "-R <str>"));
        aOpts.add(new Option("\tLexer", "L", 1, "-L <str>"));
        aOpts.add(new Option("\tFilters (default empty)",
                             "F", 1, "-F <str>[:<str>...]"));

        aOpts.addAll(DirectoryDocumentReader.listOptions());
        aOpts.addAll(SimpleLexer.listOptions());
        aOpts.addAll(LowerCaseFilter.listOptions());
        aOpts.addAll(PorterStemmer.listOptions());
        aOpts.addAll(StopWordFilter.listOptions());
        aOpts.addAll(WordLengthFilter.listOptions());

        return aOpts.elements();
    }

    public void setOptions(String[] options) throws Exception {
        Pattern patSep;
        String[] aFilters;
        Integer n;

        m_bTFIDF = Utils.getFlag('I', options);

        m_strDocReader = Utils.getOption('R', options);
        if (m_strDocReader.length() == 0)
            throw new Exception("Document reader (-R) not set.");
        else if (m_strDocReader.equals("directory"))
            m_reader = new DirectoryDocumentReader(this, options);
        else
            throw new Exception("Invalid document reader (-R).");

        m_strLexer = Utils.getOption('L', options);
        if (m_strLexer.length() == 0)
            throw new Exception("Lexer (-L) not set.");
        else if (m_strLexer.equals("simple"))
            m_lexer = new SimpleLexer(this, m_reader, options);
        else
            throw new Exception("Invalid lexer (-L).");

        m_strFilters = Utils.getOption('F', options);
        m_lstFilters = new LinkedList();
        if (m_strFilters.length() > 0) {
            patSep = Pattern.compile(":");
            aFilters = patSep.split(m_strFilters);
            for (int i = 0; i < aFilters.length; ++i)
                if (aFilters[i].length() > 0) {
                    if (aFilters[i].equals("lower_case"))
                        m_lstFilters.addLast
                            (new LowerCaseFilter(this, options));
                    else if (aFilters[i].equals("porter_stemmer"))
                        m_lstFilters.addLast
                            (new PorterStemmer(this, options));
                    else if (aFilters[i].equals("stop_word"))
                        m_lstFilters.addLast
                            (new StopWordFilter(this, options));
                    else if (aFilters[i].equals("word_length"))
                        m_lstFilters.addLast
                            (new WordLengthFilter(this, options));
                    else
                        throw new Exception
                            ("Invalid filter (-F): " + aFilters[i] + ".");
                }
        }
    }

    public String[] getOptions() {
        ArrayList aOpts;
        ListIterator it;
        TokenFilter filter;
        String[] array;

        aOpts = new ArrayList();
        if (m_bTFIDF)
            aOpts.add("-I");
        aOpts.add("-R");
        aOpts.add(m_strDocReader);
        aOpts.addAll(m_reader.getOptions());
        aOpts.add("-L");
        aOpts.add(m_strLexer);
        aOpts.addAll(m_lexer.getOptions());
        if (m_strFilters.length() > 0) {
            aOpts.add("-F");
            aOpts.add(m_strFilters);
            for (it = m_lstFilters.listIterator(); it.hasNext(); ) {
                filter = (TokenFilter) it.next();
                aOpts.addAll(filter.getOptions());
            }
        }
        array = new String[aOpts.size()];
        return (String[]) aOpts.toArray(array);
    }

    public Instances defineDataFormat() throws Exception {
        m_bFormatDefined = true;
        readInstances();
        return m_table.makeDataFormat();
    }

    public Instance generateExample() throws Exception {
        if (!m_bFormatDefined)
            throw new Exception("Dataset format not defined.");
        return m_table.getNextInstance();
    }

    public Instances generateExamples() throws Exception {
        throw new Exception("Only single mode supported.");
    }

    public String generateFinished() throws Exception {
        return "";
    }

    public boolean getSingleModeFlag() throws Exception {
        return true;
    }

    public static void main(String[] argv) throws Exception {
        Generator.makeData(new TextSource(), argv);
    }

}

