package weka.datagenerators;

import weka.datagenerators.TextSource.Real;

import weka.core.Option;
import weka.core.Utils;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * Reads a collection of text documents from a set of directories.
 * In each directory there are a number of text files, and each text
 * file contains a single document.  The name of the directory
 * specifies the class of all the documents that it contains.
 *
 * <p><b>WEKA options:</b>
 * <ul>
 *   <li><code>-D &lt;str&gt;</code> - The pathname of the base
 *   directory of the subdirectories.  This parameter has no default
 *   value and is not optional.
 *
 *   <li><code>-u &lt;str&gt;</code> - The regular expression for
 *   choosing subdirectories.  Only those subdirectories whose name
 *   matches this expression will be searched.  The substring captured
 *   by group 1 will be used as the document class, or the whole
 *   subdirectory name will be used if there are no capturing groups
 *   in the regular expression.  For example, if there are three
 *   subdirectories, namely <code>etc</code>,
 *   <code>news:comp.lang.c</code> and
 *   <code>news:comp.lang.c++</code>, and we use
 *   <code>news:(.+)</code> as the mask, then only
 *   <code>news:comp.lang.c</code> and <code>news:comp.lang.c++</code>
 *   will be searched, and the document classes that they represent
 *   will be <code>comp.lang.c</code> and <code>comp.lang.c++</code>
 *   respectively.  If we omit the parentheses inside the mask, then
 *   the document classes will be <code>news:comp.lang.c</code> and
 *   <code>news:comp.lang.c++</code> instead.  By default the mask is
 *   <code>.*</code>.
 *
 *   <li><code>-l &lt;str&gt;</code> - The regular
 *   expression for choosing text files.  Only those files whose name
 *   matches this expression will be read.  By default the mask is
 *   <code>.*</code>.
 * </ul>
 *
 * @author ywwong
 * @version $Id: DirectoryDocumentReader.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
class DirectoryDocumentReader implements DocumentReader {

    /** A file filter that accepts only readable directories. */
    protected class AcceptsDirectories implements FileFilter {
        private TextSource m_ts;
        private Pattern m_p;
        public AcceptsDirectories(TextSource ts, Pattern p) {
            m_ts = ts;
            m_p = p;
        }
        public boolean accept(File f) {
            Matcher m;
            Real d;

            if (!f.canRead() || !f.isDirectory())
                return false;
            m = m_p.matcher(f.getName());
            if (!m.matches())
                return false;
            if (m.groupCount() > 0)
                d = m_ts.registerClass(m.group(1));
            else
                d = m_ts.registerClass(m.group());
            m_aClassIndices.add(d);
            return true;
        }
    }

    /** A file filter that accepts only readable files. */
    protected class AcceptsFiles implements FileFilter {
        private Pattern m_p;
        public AcceptsFiles(Pattern p) { m_p = p; }
        public boolean accept(File f) {
            Matcher m;

            if (!f.canRead() || !f.isFile())
                return false;
            m = m_p.matcher(f.getName());
            return m.matches();
        }
    }

    /** The list of all class indices. */
    protected ArrayList m_aClassIndices;

    /** The list of all files to be read. */
    protected ArrayList m_aFiles;

    /** A list of class indices that the files belong. */
    protected ArrayList m_aClasses;

    /** The index of the next file to be read. */
    protected int m_nFile;

    /** The file being read. */
    protected BufferedInputStream m_reader;

    ////// WEKA specific. //////

    /** The option string for directory. */
    protected String m_strDir;

    /** The option string for subdirectory mask. */
    protected String m_strSubdirMask;

    /** The option string for file mask. */
    protected String m_strFileMask;

    ////// Ends WEKA specific. //////

    /**
     * Creates a directory document reader.
     *
     * @param ts  The TextSource object.
     */
    public DirectoryDocumentReader(TextSource ts, String[] options)
        throws Exception {
        super();

        File fDir;
        File[] aSubdirs;
        File[] aFiles;
        Pattern patSubdirMask;
        Pattern patFileMask;
        AcceptsDirectories adFilter;
        AcceptsFiles afFilter;

        ////// WEKA specific. //////

        m_strDir = Utils.getOption('D', options);
        if (m_strDir.length() == 0)
            throw new Exception("Base directory (-D) not set.");
        m_strSubdirMask = Utils.getOption('u', options);
        if (m_strSubdirMask.length() == 0)
            m_strSubdirMask = ".*";
        m_strFileMask = Utils.getOption('l', options);
        if (m_strFileMask.length() == 0)
            m_strFileMask = ".*";

        ////// Ends WEKA specific. //////

        m_aClassIndices = new ArrayList();
        m_aFiles = new ArrayList();
        m_aClasses = new ArrayList();
        patSubdirMask = Pattern.compile(m_strSubdirMask);
        patFileMask = Pattern.compile(m_strFileMask);
        adFilter = new AcceptsDirectories(ts, patSubdirMask);
        afFilter = new AcceptsFiles(patFileMask);

        fDir = new File(m_strDir);
        aSubdirs = fDir.listFiles(adFilter);
        for (int i = 0; i < aSubdirs.length; ++i) {
            aFiles = aSubdirs[i].listFiles(afFilter);
            for (int j = 0; j < aFiles.length; ++j) {
                m_aFiles.add(aFiles[j]);
                m_aClasses.add(new Integer(i));
            }
        }
        m_nFile = 0;
        m_reader = null;
    }

    /**
     * Tests if there are any unread documents.
     *
     * @return <code>true</code> if there are unread documents;
     * <code>false</code> if otherwise.
     */
    public boolean hasNextDocument() {
        return m_nFile < m_aFiles.size();
    }

    /**
     * Resets the reader so that when <code>read()</code> is called,
     * the next document is read.
     *
     * @return The class index of the next document; or
     * <code>null</code> if there are no more documents.
     */
    public Real nextDocument() throws IOException {
        File f;
        int nClass;

        if (m_nFile >= m_aFiles.size())
            return null;

        f = (File) m_aFiles.get(m_nFile);
        m_reader = new BufferedInputStream(new FileInputStream(f));
        nClass = ((Integer) m_aClasses.get(m_nFile)).intValue();
        ++m_nFile;
        return (Real) m_aClassIndices.get(nClass);
    }

    /**
     * Reads the next character from the current document.
     *
     * @return The next character; or -1 if
     * <code>nextDocument()</code> has never been called, or there are
     * no more characters for the current document.
     */
    public int read() throws IOException {
        int c;

        if (m_reader == null)
            return -1;

        c = m_reader.read();
        if (c == -1)
            m_reader = null;
        return c;
    }

    ////// WEKA specific. //////

    public static Collection listOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        aOpts.add(new Option("\tDirectoryDocumentReader: Base directory",
                             "D", 1, "-D <str>"));
        aOpts.add(new Option("\tDirectoryDocumentReader: Subdir mask " +
                             "(default .*)",
                             "u", 1, "-u <str>"));
        aOpts.add(new Option("\tDirectoryDocumentReader: File mask " +
                             "(default .*)",
                             "l", 1, "-l <str>"));
        return aOpts;
    }

    public Collection getOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        aOpts.add("-D");
        aOpts.add(m_strDir);
        aOpts.add("-u");
        aOpts.add(m_strSubdirMask);
        aOpts.add("-l");
        aOpts.add(m_strFileMask);
        return aOpts;
    }

}
