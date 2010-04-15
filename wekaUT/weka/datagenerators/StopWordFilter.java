package weka.datagenerators;

import weka.core.Option;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tosses words that are found in the stop list.  By default, it
 * borrows the stop list in the SMART information retrieval system.
 * The user may define his own stop list, or append stop words to the
 * built-in list.
 *
 * <p><b>WEKA options:</b>
 * <ul>
 *   <li><code>-w</code> - Whether input is lowercase only.  If this
 *   flag is set, then string comparison will be case-sensitive, which
 *   is somewhat less costly than case-insensitive comparison.  By
 *   default it is unset.
 *
 *   <li><code>-e</code> - Whether the default SMART stop list is
 *   <b>skipped</b>.  By default it is unset.
 *
 *   <li><code>-f &lt;str&gt;[:&lt;str&gt;...]</code> - A
 *   colon-separated list of stop list files.  In those files, stop
 *   words are listed in separate lines.  If there are multiple words
 *   on the same line, then only the first word will be read.  Stop
 *   words are converted to lowercase before being used.  Leading and
 *   trailing whitespace and empty lines are ignored.  If
 *   <code>filter.stop_word.use_default</code> is true, then those
 *   stop words will be appended to the default SMART stop list.  By
 *   default the list is empty.
 * </ul>
 *
 * @author ywwong
 * @version $Id: StopWordFilter.java,v 1.1.1.1 2003/01/22 07:48:27 mbilenko Exp $
 */
class StopWordFilter implements TokenFilter {

    /** The default SMART stop list. */
    protected static final String[] m_aDefStopList = {
        "a",
        "a's",
        "able",
        "about",
        "above",
        "according",
        "accordingly",
        "across",
        "actually",
        "after",
        "afterwards",
        "again",
        "against",
        "ain't",
        "all",
        "allow",
        "allows",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "an",
        "and",
        "another",
        "any",
        "anybody",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anyways",
        "anywhere",
        "apart",
        "appear",
        "appreciate",
        "appropriate",
        "are",
        "aren't",
        "around",
        "as",
        "aside",
        "ask",
        "asking",
        "associated",
        "at",
        "available",
        "away",
        "awfully",
        "b",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "believe",
        "below",
        "beside",
        "besides",
        "best",
        "better",
        "between",
        "beyond",
        "both",
        "brief",
        "but",
        "by",
        "c",
        "c'mon",
        "c's",
        "came",
        "can",
        "can't",
        "cannot",
        "cant",
        "cause",
        "causes",
        "certain",
        "certainly",
        "changes",
        "clearly",
        "co",
        "com",
        "come",
        "comes",
        "concerning",
        "consequently",
        "consider",
        "considering",
        "contain",
        "containing",
        "contains",
        "corresponding",
        "could",
        "couldn't",
        "course",
        "currently",
        "d",
        "definitely",
        "described",
        "despite",
        "did",
        "didn't",
        "different",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "done",
        "down",
        "downwards",
        "during",
        "e",
        "each",
        "edu",
        "eg",
        "eight",
        "either",
        "else",
        "elsewhere",
        "enough",
        "entirely",
        "especially",
        "et",
        "etc",
        "even",
        "ever",
        "every",
        "everybody",
        "everyone",
        "everything",
        "everywhere",
        "ex",
        "exactly",
        "example",
        "except",
        "f",
        "far",
        "few",
        "fifth",
        "first",
        "five",
        "followed",
        "following",
        "follows",
        "for",
        "former",
        "formerly",
        "forth",
        "four",
        "from",
        "further",
        "furthermore",
        "g",
        "get",
        "gets",
        "getting",
        "given",
        "gives",
        "go",
        "goes",
        "going",
        "gone",
        "got",
        "gotten",
        "greetings",
        "h",
        "had",
        "hadn't",
        "happens",
        "hardly",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he's",
        "hello",
        "help",
        "hence",
        "her",
        "here",
        "here's",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "hi",
        "him",
        "himself",
        "his",
        "hither",
        "hopefully",
        "how",
        "howbeit",
        "however",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "ie",
        "if",
        "ignored",
        "immediate",
        "in",
        "inasmuch",
        "inc",
        "indeed",
        "indicate",
        "indicated",
        "indicates",
        "inner",
        "insofar",
        "instead",
        "into",
        "inward",
        "is",
        "isn't",
        "it",
        "it'd",
        "it'll",
        "it's",
        "its",
        "itself",
        "j",
        "just",
        "k",
        "keep",
        "keeps",
        "kept",
        "know",
        "knows",
        "known",
        "l",
        "last",
        "lately",
        "later",
        "latter",
        "latterly",
        "least",
        "less",
        "lest",
        "let",
        "let's",
        "like",
        "liked",
        "likely",
        "little",
        "look",
        "looking",
        "looks",
        "ltd",
        "m",
        "mainly",
        "many",
        "may",
        "maybe",
        "me",
        "mean",
        "meanwhile",
        "merely",
        "might",
        "more",
        "moreover",
        "most",
        "mostly",
        "much",
        "must",
        "my",
        "myself",
        "n",
        "name",
        "namely",
        "nd",
        "near",
        "nearly",
        "necessary",
        "need",
        "needs",
        "neither",
        "never",
        "nevertheless",
        "new",
        "next",
        "nine",
        "no",
        "nobody",
        "non",
        "none",
        "noone",
        "nor",
        "normally",
        "not",
        "nothing",
        "novel",
        "now",
        "nowhere",
        "o",
        "obviously",
        "of",
        "off",
        "often",
        "oh",
        "ok",
        "okay",
        "old",
        "on",
        "once",
        "one",
        "ones",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "outside",
        "over",
        "overall",
        "own",
        "p",
        "particular",
        "particularly",
        "per",
        "perhaps",
        "placed",
        "please",
        "plus",
        "possible",
        "presumably",
        "probably",
        "provides",
        "q",
        "que",
        "quite",
        "qv",
        "r",
        "rather",
        "rd",
        "re",
        "really",
        "reasonably",
        "regarding",
        "regardless",
        "regards",
        "relatively",
        "respectively",
        "right",
        "s",
        "said",
        "same",
        "saw",
        "say",
        "saying",
        "says",
        "second",
        "secondly",
        "see",
        "seeing",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "seen",
        "self",
        "selves",
        "sensible",
        "sent",
        "serious",
        "seriously",
        "seven",
        "several",
        "shall",
        "she",
        "should",
        "shouldn't",
        "since",
        "six",
        "so",
        "some",
        "somebody",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhat",
        "somewhere",
        "soon",
        "sorry",
        "specified",
        "specify",
        "specifying",
        "still",
        "sub",
        "such",
        "sup",
        "sure",
        "t",
        "t's",
        "take",
        "taken",
        "tell",
        "tends",
        "th",
        "than",
        "thank",
        "thanks",
        "thanx",
        "that",
        "that's",
        "thats",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "there's",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "theres",
        "thereupon",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "think",
        "third",
        "this",
        "thorough",
        "thoroughly",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "took",
        "toward",
        "towards",
        "tried",
        "tries",
        "truly",
        "try",
        "trying",
        "twice",
        "two",
        "u",
        "un",
        "under",
        "unfortunately",
        "unless",
        "unlikely",
        "until",
        "unto",
        "up",
        "upon",
        "us",
        "use",
        "used",
        "useful",
        "uses",
        "using",
        "usually",
        "uucp",
        "v",
        "value",
        "various",
        "very",
        "via",
        "viz",
        "vs",
        "w",
        "want",
        "wants",
        "was",
        "wasn't",
        "way",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "welcome",
        "well",
        "went",
        "were",
        "weren't",
        "what",
        "what's",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "where's",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "who's",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "willing",
        "wish",
        "with",
        "within",
        "without",
        "won't",
        "wonder",
        "would",
        "wouldn't",
        "x",
        "y",
        "yes",
        "yet",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "z",
        "zero",
    };

    /** The stop list. */
    protected HashSet m_aStopList;

    /** True if input is lowercase only. */
    protected boolean m_bLowerCaseOnly;

    ////// WEKA specific. //////

    /** True if the default stop list is skipped. */
    protected boolean m_bSkipDefault;

    /** The option string for stop list files. */
    protected String m_strFiles;

    ////// Ends WEKA specific. //////

    /**
     * Creates a stop word filter.
     *
     * @param ts  The TextSource object.
     */
    public StopWordFilter(TextSource ts, String[] options) throws Exception {
        Pattern patSep;
        String[] aFiles;

        m_aStopList = new HashSet();

        ////// WEKA specific. //////

        m_bSkipDefault = Utils.getFlag('e', options);
        if (!m_bSkipDefault)
            for (int i = 0; i < m_aDefStopList.length; ++i)
                m_aStopList.add(m_aDefStopList[i]);

        m_strFiles = Utils.getOption('f', options);
        if (m_strFiles.length() > 0) {
            patSep = Pattern.compile(":");
            aFiles = patSep.split(m_strFiles);
            for (int i = 0; i < aFiles.length; ++i)
                addFile(aFiles[i]);
        }

        m_bLowerCaseOnly = Utils.getFlag('w', options);
    }

    protected Pattern m_patChomp = null;

    /**
     * Add the words specified in a given file to the stop list.
     * Stop words are listed in separate lines.  If there are multiple
     * words on the same line, then only the first word will be read.
     * Stop words are converted to lowercase before being put in the
     * list.  Leading and trailing whitespace and empty lines are
     * ignored.
     *
     * @param strFileName  The name of the stop list file.
     */
    protected void addFile(String strFileName) throws IOException {
        BufferedReader in;
        String str;
        Matcher matChomp;

        if (m_patChomp == null)
            m_patChomp = Pattern.compile("^\\s*(\\S*)");

        in = new BufferedReader(new FileReader(strFileName));
        str = in.readLine();
        while (str != null) {
            matChomp = m_patChomp.matcher(str);
            matChomp.lookingAt();
            str = matChomp.group(1);
            if (str.length() > 0)
                m_aStopList.add(str.toLowerCase());
            str = in.readLine();
        }
    }

    /**
     * Tosses tokens that appear in the stop list.  Case is ignored
     * when doing comparison.
     *
     * @param strToken  The input token
     * @return <code>null</code> if the input token appears in the
     * stop list; the token itself if otherwise.
     */
    public String apply(String strToken) {
        if (m_bLowerCaseOnly) {
            if (m_aStopList.contains(strToken))
                return null;
        } else {
            if (m_aStopList.contains(strToken.toLowerCase()))
                return null;
        }
        return strToken;
    }

    ////// WEKA specific. //////

    public static Collection listOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        aOpts.add(new Option("\tStopWordFilter: " +
                             "Set if input is guanranteed lowercase",
                             "w", 0, "-w"));
        aOpts.add(new Option("\tStopWordFilter: " +
                             "Skip default SMART stop list",
                             "e", 0, "-e"));
        aOpts.add(new Option("\tStopWordFilter: Stop list files " +
                             "(default empty)",
                             "f", 1, "-f <str>[:<str>...]"));
        return aOpts;
    }

    public Collection getOptions() {
        ArrayList aOpts;

        aOpts = new ArrayList();
        if (m_bLowerCaseOnly)
            aOpts.add("-w"); // ??
        if (m_bSkipDefault)
            aOpts.add("-e"); // ??
        if (m_strFiles.length() > 0) {
            aOpts.add("-f");
            aOpts.add(m_strFiles);
        }
        return aOpts;
    }

}
