/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    ResultsPanel.java
 *    Copyright (C) 1999 Len Trigg
 *    Modified by Prem Melville
 *
 */


package weka.gui.experiment;

import java.util.*;
import weka.gui.ExtensionFileFilter;
import weka.gui.ListSelectorDialog;
import weka.gui.ResultHistoryPanel;
import weka.gui.SaveBuffer;
import weka.experiment.Experiment;
import weka.experiment.InstancesResultListener;
import weka.experiment.DatabaseResultListener;
import weka.experiment.PairedTTester;
import weka.experiment.InstanceQuery;
import weka.core.Utils;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Range;
import weka.core.Instance;

//=============== BEGIN EDIT mbilenko ===============
import java.io.PrintWriter;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
//=============== END EDIT mbilenko ===============

import java.io.Reader;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.File;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.JFrame;
import javax.swing.JButton;
import javax.swing.JTextArea;
import javax.swing.BorderFactory;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;
import javax.swing.filechooser.FileFilter;
import javax.swing.JFileChooser;
import javax.swing.JComboBox;
import javax.swing.JTextField;
import javax.swing.DefaultComboBoxModel;
import javax.swing.DefaultListModel;
import javax.swing.JList;
import javax.swing.ListSelectionModel;
import javax.swing.JOptionPane;
import javax.swing.JCheckBox;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.awt.Dimension;
import javax.swing.SwingUtilities;


/** 
 * This panel controls simple analysis of experimental results.
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 1.12 $
 */
public class ResultsPanel extends JPanel {

  /** Message shown when no experimental results have been loaded */
  protected static final String NO_SOURCE = "No source";

  /** Click to load results from a file */
  protected JButton m_FromFileBut = new JButton("File...");

  /** Click to load results from a database */
  protected JButton m_FromDBaseBut = new JButton("Database...");

  /** Click to get results from the destination given in the experiment */
  protected JButton m_FromExpBut = new JButton("Experiment");

  /** Displays a message about the current result set */
  protected JLabel m_FromLab = new JLabel(NO_SOURCE);

  /**
   * This is needed to get around a bug in Swing <= 1.1 -- Once everyone
   * is using Swing 1.1.1 or higher just remove this variable and use the
   * no-arg constructor to DefaultComboBoxModel
   */
  private static String [] FOR_JFC_1_1_DCBM_BUG = {""};

  /** The model embedded in m_DatasetCombo */
  protected DefaultComboBoxModel m_DatasetModel =
    new DefaultComboBoxModel(FOR_JFC_1_1_DCBM_BUG);
  
  /** The model embedded in m_RunCombo */
  protected DefaultComboBoxModel m_RunModel =
    new DefaultComboBoxModel(FOR_JFC_1_1_DCBM_BUG);
  
  /** The model embedded in m_CompareCombo */
  protected DefaultComboBoxModel m_CompareModel = 
    new DefaultComboBoxModel(FOR_JFC_1_1_DCBM_BUG);
  
  /** The model embedded in m_TestsList */
  protected DefaultListModel m_TestsModel = new DefaultListModel();

  /** Displays the currently selected column names for the scheme & options */
  protected JLabel m_DatasetKeyLabel = new JLabel("Row key fields",
						 SwingConstants.RIGHT);

  /** Click to edit the columns used to determine the scheme */
  protected JButton m_DatasetKeyBut = new JButton("Select keys...");

  /** Stores the list of attributes for selecting the scheme columns */
  protected DefaultListModel m_DatasetKeyModel = new DefaultListModel();

  /** Displays the list of selected columns determining the scheme */
  protected JList m_DatasetKeyList = new JList(m_DatasetKeyModel);

  /** Lets the user select which column contains the run number */
  protected JComboBox m_RunCombo = new JComboBox(m_RunModel);

  /** Displays the currently selected column names for the scheme & options */
  protected JLabel m_ResultKeyLabel = new JLabel("Column key fields",
						 SwingConstants.RIGHT);

  /** Click to edit the columns used to determine the scheme */
  protected JButton m_ResultKeyBut = new JButton("Select keys...");

  /** Stores the list of attributes for selecting the scheme columns */
  protected DefaultListModel m_ResultKeyModel = new DefaultListModel();

  /** Displays the list of selected columns determining the scheme */
  protected JList m_ResultKeyList = new JList(m_ResultKeyModel);

  /** Lets the user select which scheme to base comparisons against */
  protected JButton m_TestsButton = new JButton("Select base...");

  /** Holds the list of schemes to base the test against */
  protected JList m_TestsList = new JList(m_TestsModel);

  /** Lets the user select which performance measure to analyze */
  protected JComboBox m_CompareCombo = new JComboBox(m_CompareModel);

  /** Lets the user edit the test significance */
  protected JTextField m_SigTex = new JTextField("0.05");

  /** Lets the user select whether standard deviations are to be output
      or not */
  protected JCheckBox m_ShowStdDevs = 
    new JCheckBox("");

  /** Click to start the test */
  protected JButton m_PerformBut = new JButton("Perform test");
  
  /** Click to save test output to a file */
  protected JButton m_SaveOutBut = new JButton("Save output");

  //=============== BEGIN EDIT mbilenko ===============
  /** Click to launch gnuplot */
  protected JButton m_PlotBut = new JButton("Plot");
  //=============== END EDIT mbilenko ===============
  
    //=============== BEGIN EDIT melville ===============
    /** Lets the user specify the precision of results desired */
    protected JTextField m_PrecTex = new JTextField("2");

    //To remember index of error for computing error reductions
    protected int m_ErrorCompareCol;
    //=============== END EDIT melville ===============
  
  /** The buffer saving object for saving output */
  SaveBuffer m_SaveOut = new SaveBuffer(null, this);

  /** Displays the output of tests */
  protected JTextArea m_OutText = new JTextArea();

  /** A panel controlling results viewing */
  protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);

  /** Filter to ensure only arff files are selected for result files */  
  protected FileFilter m_ArffFilter =
    new ExtensionFileFilter(Instances.FILE_EXTENSION, "Arff data files");

  
  /** The file chooser for selecting result files */
  protected JFileChooser m_FileChooser = new JFileChooser(new File(System.getProperty("user.dir")));

  /** The PairedTTester object */
  protected PairedTTester m_TTester = new PairedTTester();
  
  /** The instances we're extracting results from */
  protected Instances m_Instances;

  /** Does any database querying for us */
  protected InstanceQuery m_InstanceQuery;

  /** A thread to load results instances from a file or database */
  protected Thread m_LoadThread;
  
  /** An experiment (used for identifying a result source) -- optional */
  protected Experiment m_Exp;

  /** An actionlisteners that updates ttest settings */
  protected ActionListener m_ConfigureListener = new ActionListener() {
    public void actionPerformed(ActionEvent e) {
      m_TTester.setRunColumn(m_RunCombo.getSelectedIndex());
      setTTester();
    }
  };
  
  private Dimension COMBO_SIZE = new Dimension(150, m_ResultKeyBut
					       .getPreferredSize().height);
  /**
   * Creates the results panel with no initial experiment.
   */
  public ResultsPanel() {

    // Create/Configure/Connect components
    m_FileChooser.setFileFilter(m_ArffFilter);
    m_FileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
    m_FromExpBut.setEnabled(false);
    m_FromExpBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	if (m_LoadThread == null) {
	  m_LoadThread = new Thread() {
	    public void run() {
	      setInstancesFromExp(m_Exp);
	      m_LoadThread = null;
	    }
	  };
	  m_LoadThread.start();
	}
      }
    });
    m_FromDBaseBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	if (m_LoadThread == null) {
	  m_LoadThread = new Thread() {
	    public void run() {
	      setInstancesFromDBaseQuery();
	    m_LoadThread = null;
	    }
	  };
	  m_LoadThread.start();
	}
      }
    });
    m_FromFileBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	int returnVal = m_FileChooser.showOpenDialog(ResultsPanel.this);
	if (returnVal == JFileChooser.APPROVE_OPTION) {
	  final File selected = m_FileChooser.getSelectedFile();
	  if (m_LoadThread == null) {
	    m_LoadThread = new Thread() {
	      public void run() {
		setInstancesFromFile(selected);
		m_LoadThread = null;
	      }
	    };
	    m_LoadThread.start();
	  }
	}
      }
    });
    setComboSizes();
    m_DatasetKeyBut.setEnabled(false);
    m_DatasetKeyBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	setDatasetKeyFromDialog();
      }
    });
    m_DatasetKeyList.setSelectionMode(ListSelectionModel
				      .MULTIPLE_INTERVAL_SELECTION);
    m_RunCombo.setEnabled(false);
    m_RunCombo.addActionListener(m_ConfigureListener);
    m_ResultKeyBut.setEnabled(false);
    m_ResultKeyBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	setResultKeyFromDialog();
      }
    });
    m_ResultKeyList.setSelectionMode(ListSelectionModel
				     .MULTIPLE_INTERVAL_SELECTION);
    m_CompareCombo.setEnabled(false);

    m_SigTex.setEnabled(false);
    m_PrecTex.setEnabled(false);
    //=============== BEGIN EDIT melville ===============
    m_TestsButton.setEnabled(false);
    //=============== END EDIT melville ===============
    m_TestsButton.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  setTestBaseFromDialog();
	}
      });

    m_PerformBut.setEnabled(false);
    m_PerformBut.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent e) {
	performTest();
	m_SaveOutBut.setEnabled(true);
	//=============== BEGIN EDIT mbilenko ===============
	m_PlotBut.setEnabled(true);
	//=============== END EDIT mbilenko ===============
      }
    });
    m_SaveOutBut.setEnabled(false);
    m_SaveOutBut.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  saveBuffer();
	}
      });
    //=============== BEGIN EDIT mbilenko ===============
    m_PlotBut.setEnabled(false);
    m_PlotBut.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  plotOutput();
	}
      });
    //=============== END EDIT mbilenko ===============
	
    
    m_OutText.setFont(new Font("Monospaced", Font.PLAIN, 12));
    m_OutText.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
    m_OutText.setEditable(false);
    m_History.setBorder(BorderFactory.createTitledBorder("Result list"));


    // Set up the GUI layout
    JPanel p1 = new JPanel();
    p1.setBorder(BorderFactory.createTitledBorder("Source"));
    JPanel p2 = new JPanel();
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints constraints = new GridBagConstraints();
    p2.setBorder(BorderFactory.createEmptyBorder(5, 5, 10, 5));
    //    p2.setLayout(new GridLayout(1, 3));
    p2.setLayout(gb);
    constraints.gridx=0;constraints.gridy=0;constraints.weightx=5;
    constraints.fill = GridBagConstraints.HORIZONTAL;
    constraints.gridwidth=1;constraints.gridheight=1;
    constraints.insets = new Insets(0,2,0,2);
    p2.add(m_FromFileBut,constraints);
    constraints.gridx=1;constraints.gridy=0;constraints.weightx=5;
    constraints.gridwidth=1;constraints.gridheight=1;
    p2.add(m_FromDBaseBut,constraints);
    constraints.gridx=2;constraints.gridy=0;constraints.weightx=5;
    constraints.gridwidth=1;constraints.gridheight=1;
    p2.add(m_FromExpBut,constraints);
    p1.setLayout(new BorderLayout());
    p1.add(m_FromLab, BorderLayout.CENTER);
    p1.add(p2, BorderLayout.EAST);

    JPanel p3 = new JPanel();
    p3.setBorder(BorderFactory.createTitledBorder("Configure test"));
    GridBagLayout gbL = new GridBagLayout();
    p3.setLayout(gbL);

    GridBagConstraints gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 0;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(m_DatasetKeyLabel,gbC);
    p3.add(m_DatasetKeyLabel);
    gbC = new GridBagConstraints();
    gbC.gridy = 0;     gbC.gridx = 1;  gbC.weightx = 100;
    gbC.insets = new Insets(5,0,5,0);
    gbL.setConstraints(m_DatasetKeyBut, gbC);
    p3.add(m_DatasetKeyBut);

    JLabel lab = new JLabel("Run field", SwingConstants.RIGHT);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 1;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(lab, gbC);
    p3.add(lab);
    gbC = new GridBagConstraints();
    gbC.gridy = 1;     gbC.gridx = 1;  gbC.weightx = 100;
    gbC.insets = new Insets(5,0,5,0);
    gbL.setConstraints(m_RunCombo, gbC);
    p3.add(m_RunCombo);
    
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 2;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(m_ResultKeyLabel, gbC);
    p3.add(m_ResultKeyLabel);
    gbC = new GridBagConstraints();
    gbC.gridy = 2;     gbC.gridx = 1;  gbC.weightx = 100;
    gbC.insets = new Insets(5,0,5,0);
    gbL.setConstraints(m_ResultKeyBut, gbC);
    p3.add(m_ResultKeyBut);
    
    lab = new JLabel("Comparison field", SwingConstants.RIGHT);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 3;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(lab, gbC);
    p3.add(lab);
    gbC = new GridBagConstraints();
    gbC.gridy = 3;     gbC.gridx = 1;  gbC.weightx = 100;
    gbC.insets = new Insets(5,0,5,0);
    gbL.setConstraints(m_CompareCombo, gbC);
    p3.add(m_CompareCombo);
    
    lab = new JLabel("Significance", SwingConstants.RIGHT);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 4;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(lab, gbC);
    p3.add(lab);
    gbC = new GridBagConstraints();
    gbC.gridy = 4;     gbC.gridx = 1;  gbC.weightx = 100;
    gbL.setConstraints(m_SigTex, gbC);
    p3.add(m_SigTex);
    
    lab = new JLabel("Test base", SwingConstants.RIGHT);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 5;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(lab, gbC);
    p3.add(lab);
    gbC = new GridBagConstraints();
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 5;     gbC.gridx = 1;  gbC.weightx = 100;
    gbC.insets = new Insets(5,0,5,0);
    gbL.setConstraints(m_TestsButton, gbC);
    p3.add(m_TestsButton);

    //=============== BEGIN EDIT melville ===============
    lab = new JLabel("Significant digits", SwingConstants.RIGHT);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 6;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(lab, gbC);
    p3.add(lab);
    gbC = new GridBagConstraints();
    gbC.gridy = 6;     gbC.gridx = 1;  gbC.weightx = 100;
    gbL.setConstraints(m_PrecTex, gbC);
    p3.add(m_PrecTex);
    //=============== END EDIT melville ===============

    lab = new JLabel("Show std. deviations", SwingConstants.RIGHT);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.EAST;
    gbC.gridy = 7;     gbC.gridx = 0;
    gbC.insets = new Insets(2, 10, 2, 10);
    gbL.setConstraints(lab, gbC);
    p3.add(lab);
    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.WEST;
    gbC.gridy = 7;     gbC.gridx = 1;  gbC.weightx = 100;
    gbC.insets = new Insets(5,0,5,0);
    gbL.setConstraints(m_ShowStdDevs, gbC);
    p3.add(m_ShowStdDevs);

    JPanel output = new JPanel();
    output.setLayout(new BorderLayout());
    output.setBorder(BorderFactory.createTitledBorder("Test output"));
    output.add(new JScrollPane(m_OutText), BorderLayout.CENTER);

    JPanel mondo = new JPanel();
    gbL = new GridBagLayout();
    mondo.setLayout(gbL);
    gbC = new GridBagConstraints();
    //    gbC.anchor = GridBagConstraints.WEST;
    //    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 0;     gbC.gridx = 0;
    gbL.setConstraints(p3, gbC);
    mondo.add(p3);

    JPanel bts = new JPanel();
    bts.setLayout(new GridLayout(1,2,5,5));
    bts.add(m_PerformBut);
    bts.add(m_SaveOutBut);
    //=============== BEGIN EDIT mbilenko ===============
    bts.add(m_PlotBut);
    //=============== END EDIT mbilenko ===============

    gbC = new GridBagConstraints();
    gbC.anchor = GridBagConstraints.NORTH;
    gbC.fill = GridBagConstraints.HORIZONTAL;
    gbC.gridy = 1;     gbC.gridx = 0;
    gbC.insets = new Insets(5,5,5,5);
    gbL.setConstraints(bts, gbC);
    mondo.add(bts);
    gbC = new GridBagConstraints();
    //gbC.anchor = GridBagConstraints.NORTH;
    gbC.fill = GridBagConstraints.BOTH;
    gbC.gridy = 2;     gbC.gridx = 0; gbC.weightx = 0;
    gbL.setConstraints(m_History, gbC);
    mondo.add(m_History);
    gbC = new GridBagConstraints();
    gbC.fill = GridBagConstraints.BOTH;
    gbC.gridy = 0;     gbC.gridx = 1;
    gbC.gridheight = 3;
    gbC.weightx = 100; gbC.weighty = 100;
    gbL.setConstraints(output, gbC);
    mondo.add(output);

    setLayout(new BorderLayout());
    add(p1, BorderLayout.NORTH);
    add(mondo , BorderLayout.CENTER);
  }

  /**
   * Sets the combo-boxes to a fixed size so they don't take up too much room
   * that would be better devoted to the test output box
   */
  protected void setComboSizes() {
    
    m_DatasetKeyBut.setPreferredSize(COMBO_SIZE);
    m_RunCombo.setPreferredSize(COMBO_SIZE);
    m_ResultKeyBut.setPreferredSize(COMBO_SIZE);
    m_CompareCombo.setPreferredSize(COMBO_SIZE);
    m_SigTex.setPreferredSize(COMBO_SIZE);
    
    m_DatasetKeyBut.setMaximumSize(COMBO_SIZE);
    m_RunCombo.setMaximumSize(COMBO_SIZE);
    m_ResultKeyBut.setMaximumSize(COMBO_SIZE);
    m_CompareCombo.setMaximumSize(COMBO_SIZE);
    m_SigTex.setMaximumSize(COMBO_SIZE);
    
    m_DatasetKeyBut.setMinimumSize(COMBO_SIZE);
    m_RunCombo.setMinimumSize(COMBO_SIZE);
    m_ResultKeyBut.setMinimumSize(COMBO_SIZE);
    m_CompareCombo.setMinimumSize(COMBO_SIZE);
    m_SigTex.setMinimumSize(COMBO_SIZE);
    
    //=============== BEGIN EDIT melville ===============
    m_PrecTex.setPreferredSize(COMBO_SIZE);
    m_PrecTex.setMaximumSize(COMBO_SIZE);
    m_PrecTex.setMinimumSize(COMBO_SIZE);
    //=============== END EDIT melville ===============
  }
  
  /**
   * Tells the panel to use a new experiment.
   *
   * @param exp a value of type 'Experiment'
   */
  public void setExperiment(Experiment exp) {
    
    m_Exp = exp;
    setFromExpEnabled();
  }

  /**
   * Updates whether the current experiment is of a type that we can
   * determine the results destination.
   */
  protected void setFromExpEnabled() {

    if ((m_Exp.getResultListener() instanceof InstancesResultListener)
	|| (m_Exp.getResultListener() instanceof DatabaseResultListener)) {
      m_FromExpBut.setEnabled(true);
    } else {
      m_FromExpBut.setEnabled(false);
    }
  }

  /**
   * Queries the user enough to make a database query to retrieve experiment
   * results.
   */
  protected void setInstancesFromDBaseQuery() {

    try {
      if (m_InstanceQuery == null) {
	m_InstanceQuery = new InstanceQuery();
      }
      String dbaseURL = m_InstanceQuery.getDatabaseURL();
      dbaseURL = (String) JOptionPane.showInputDialog(this,
					     "Enter the database URL",
					     "Query Database",
					     JOptionPane.PLAIN_MESSAGE,
					     null,
					     null,
					     dbaseURL);
      if (dbaseURL == null) {
	m_FromLab.setText("Cancelled");
	return;
      }
      m_InstanceQuery.setDatabaseURL(dbaseURL);
      m_InstanceQuery.connectToDatabase();
      if (!m_InstanceQuery.experimentIndexExists()) {
	m_FromLab.setText("No experiment index");
	return;
      }
      m_FromLab.setText("Getting experiment index");
      Instances index = m_InstanceQuery.retrieveInstances("SELECT * FROM "
				       + InstanceQuery.EXP_INDEX_TABLE);
      if (index.numInstances() == 0) {
	m_FromLab.setText("No experiments available");
	return;	
      }
      m_FromLab.setText("Got experiment index");

      DefaultListModel lm = new DefaultListModel();
      for (int i = 0; i < index.numInstances(); i++) {
	lm.addElement(index.instance(i).toString());
      }
      JList jl = new JList(lm);
      ListSelectorDialog jd = new ListSelectorDialog(null, jl);
      int result = jd.showDialog();
      if (result != ListSelectorDialog.APPROVE_OPTION) {
	m_FromLab.setText("Cancelled");
	return;
      }
      Instance selInst = index.instance(jl.getSelectedIndex());
      Attribute tableAttr = index.attribute(InstanceQuery.EXP_RESULT_COL);
      String table = InstanceQuery.EXP_RESULT_PREFIX
	+ selInst.toString(tableAttr);

      setInstancesFromDatabaseTable(table);
    } catch (Exception ex) {
      m_FromLab.setText("Problem reading database");
    }
  }
  
  /**
   * Examines the supplied experiment to determine the results destination
   * and attempts to load the results.
   *
   * @param exp a value of type 'Experiment'
   */
  protected void setInstancesFromExp(Experiment exp) {

    if (exp.getResultListener() instanceof InstancesResultListener) {
      File resultFile = ((InstancesResultListener) exp.getResultListener())
	.getOutputFile();
      if ((resultFile == null) || (resultFile.getName().equals("-"))) {
	m_FromLab.setText("No result file");
      } else {
	setInstancesFromFile(resultFile);
      }
    } else if (exp.getResultListener() instanceof DatabaseResultListener) {
      String dbaseURL = ((DatabaseResultListener) exp.getResultListener())
	.getDatabaseURL();
      try {
	if (m_InstanceQuery == null) {
	  m_InstanceQuery = new InstanceQuery();
	}
	m_InstanceQuery.setDatabaseURL(dbaseURL);
	m_InstanceQuery.connectToDatabase();
	String tableName = m_InstanceQuery
	  .getResultsTableName(exp.getResultProducer());
	setInstancesFromDatabaseTable(tableName);
      } catch (Exception ex) {
	m_FromLab.setText("Problem reading database");
      }
    } else {
      m_FromLab.setText("Can't get results from experiment");
    }
  }

  
  /**
   * Queries a database to load results from the specified table name. The
   * database connection must have already made by m_InstanceQuery.
   *
   * @param tableName the name of the table containing results to retrieve.
   */
  protected void setInstancesFromDatabaseTable(String tableName) {

    try {
      m_FromLab.setText("Reading from database, please wait...");
      final Instances i = m_InstanceQuery.retrieveInstances("SELECT * FROM "
						      + tableName);
      SwingUtilities.invokeAndWait(new Runnable() {
	public void run() {
	  setInstances(i);
	}
      });
      m_InstanceQuery.disconnectFromDatabase();
    } catch (Exception ex) {
      m_FromLab.setText(ex.getMessage());
    }
  }
  
  /**
   * Loads results from a set of instances contained in the supplied
   * file.
   *
   * @param f a value of type 'File'
   */
  protected void setInstancesFromFile(File f) {
      
    try {
      m_FromLab.setText("Reading from file...");
      Reader r = new BufferedReader(new FileReader(f));
      setInstances(new Instances(r));
    } catch (Exception ex) {
      ex.printStackTrace();
      m_FromLab.setText(ex.getMessage());
    }
  }

  /**
   * Sets up the panel with a new set of instances, attempting
   * to guess the correct settings for various columns.
   *
   * @param newInstances the new set of results.
   */
  public void setInstances(Instances newInstances) {

    m_Instances = newInstances;
    m_TTester.setInstances(m_Instances);
    m_FromLab.setText("Got " + m_Instances.numInstances() + " results");

    // Temporarily remove the configuration listener
    m_RunCombo.removeActionListener(m_ConfigureListener);
    
    // Do other stuff
    m_DatasetKeyModel.removeAllElements();
    m_RunModel.removeAllElements();
    m_ResultKeyModel.removeAllElements();
    m_CompareModel.removeAllElements();
    int datasetCol = -1;
    int runCol = -1;
    String selectedList = "";
    String selectedListDataset = "";
      //=============== BEGIN EDIT melville ===============
    boolean noise = false;//keep track of whether noise levels eval is required
    boolean learning = false;//keep track of whether learning curve eval is required
    boolean fraction = false;//keep track of whether fractions of datasets are provided for learning
    //the key on which to base the learning curves (either total instances or fraction) 
    int learning_key = -1;
    boolean classificationTask = false;//used to determine if regression measures should be selected
      //=============== END EDIT melville ===============
    for (int i = 0; i < m_Instances.numAttributes(); i++) {
      String name = m_Instances.attribute(i).name();
      m_DatasetKeyModel.addElement(name);
      m_RunModel.addElement(name);
      m_ResultKeyModel.addElement(name);
      m_CompareModel.addElement(name);

            //=============== BEGIN EDIT melville ===============
      //If learning curves were generated then treat each
      //dataset + pt combination as a different dataset
      if(name.toLowerCase().equals("key_noise_levels")){
	  //noise overrides learning curves - but treat noise levels
	  //like pts on learning curve
	  learning_key = i;
	  learning = true;
	  noise = true;
	  //fraction = true;
      }else if(name.toLowerCase().equals("key_fraction_instances") && !noise){
	  //fraction overrides total_instances
	  learning_key = i;
	  learning = true;
	  fraction = true;
      }else if(name.toLowerCase().equals("key_total_instances") && !learning){
	  learning_key = i;
	  learning = true;
      }else
            //=============== END EDIT melville ===============	
      if (name.toLowerCase().equals("key_dataset")) {
	m_DatasetKeyList.addSelectionInterval(i, i);
	selectedListDataset += "," + (i + 1);
      } else if ((runCol == -1)
		 && (name.toLowerCase().equals("key_run"))) {
	m_RunCombo.setSelectedIndex(i);
	runCol = i;
      } else if (name.toLowerCase().equals("key_scheme") ||
		 name.toLowerCase().equals("key_scheme_options") ||
		 name.toLowerCase().equals("key_scheme_version_id")) {
	m_ResultKeyList.addSelectionInterval(i, i);
	selectedList += "," + (i + 1);
	//=============== BEGIN EDIT mbilenko ===============
      // automatic selection of the correct measure for clustering experiments 
      } else if (name.toLowerCase().indexOf("pairwise_f_measure") != -1) {
	  m_CompareCombo.setSelectedIndex(i);
	  m_ErrorCompareCol = i;
      }
      // automatic selection of the correct measure for deduping experiments 
       else if (name.toLowerCase().equals("precision")) {
	m_CompareCombo.setSelectedIndex(i);
      //=============== END EDIT mbilenko ===============	
       }else if (name.toLowerCase().indexOf("percent_correct") != -1) {
	  m_CompareCombo.setSelectedIndex(i);
	  classificationTask=true;
       }else if (!classificationTask && (name.toLowerCase().indexOf("root_mean_squared_error") != -1)) {
	   // automatic selection of the correct measure for regression experiments 
	   m_CompareCombo.setSelectedIndex(i);
       }else if (name.toLowerCase().indexOf("percent_incorrect") != -1) {
	   m_ErrorCompareCol = i;
	   //remember index of error for computing error reductions
      }
    }
      //=============== BEGIN EDIT melville ===============	
    if(learning){
	m_DatasetKeyList.addSelectionInterval(learning_key, learning_key);
	selectedListDataset += "," + (learning_key + 1);
	m_CompareModel.addElement("%Error_reduction");
	m_CompareModel.addElement("Top_20%_%Error_reduction");
    }
      //=============== END EDIT melville ===============	
    
    if (runCol == -1) {
      runCol = 0;
    }
    m_DatasetKeyBut.setEnabled(true);
    m_RunCombo.setEnabled(true);
    m_ResultKeyBut.setEnabled(true);
    m_CompareCombo.setEnabled(true);
    
    // Reconnect the configuration listener
    m_RunCombo.addActionListener(m_ConfigureListener);
    
    // Set up the TTester with the new data
    m_TTester.setRunColumn(runCol);
    Range generatorRange = new Range();
    if (selectedList.length() != 0) {
      try {
	generatorRange.setRanges(selectedList);
      } catch (Exception ex) {
	ex.printStackTrace();
	System.err.println(ex.getMessage());
      }
    }
    m_TTester.setResultsetKeyColumns(generatorRange);

    generatorRange = new Range();
    if (selectedListDataset.length() != 0) {
      try {
	generatorRange.setRanges(selectedListDataset);
      } catch (Exception ex) {
	ex.printStackTrace();
	System.err.println(ex.getMessage());
      }
    }
    m_TTester.setDatasetKeyColumns(generatorRange);
    //=============== BEGIN EDIT melville ===============
    m_TTester.setLearningCurve(learning);
    m_TTester.setFraction(fraction);
    if(learning){//get points on the learning curve
	Attribute attr;
	if(noise){//override fraction
	    attr = m_Instances.attribute("Key_Noise_levels");
	}else if(fraction){
	    attr = m_Instances.attribute("Key_Fraction_instances");
	}else {
	    attr = m_Instances.attribute("Key_Total_instances");
	}
	double []pts = new double [attr.numValues()];
	for(int k=0; k<attr.numValues(); k++){
	    pts[k] = Double.parseDouble(attr.value(k));
	}
	//sort points
	Arrays.sort(pts);
	m_TTester.setPoints(pts);
    }
    //=============== END EDIT melville ===============
    m_SigTex.setEnabled(true);
    m_PrecTex.setEnabled(true);

    setTTester();
  }

  /**
   * Updates the test chooser with possible tests
   */
  protected void setTTester() {
    
    String name = (new SimpleDateFormat("HH:mm:ss - "))
      .format(new Date())
      + "Available resultsets";
    StringBuffer outBuff = new StringBuffer();
    outBuff.append("Available resultsets\n"
		   + m_TTester.resultsetKey() + "\n\n");
    m_History.addResult(name, outBuff);
    m_History.setSingle(name);

    m_TestsModel.removeAllElements();
    for (int i = 0; i < m_TTester.getNumResultsets(); i++) {
      String tname = m_TTester.getResultsetName(i);
      /*      if (tname.length() > 20) {
	tname = tname.substring(0, 20);
	} */
      m_TestsModel.addElement(tname);
    }
    m_TestsModel.addElement("Summary");
    m_TestsModel.addElement("Ranking");
    //================ BEGIN EDIT melville ================
    m_TestsModel.addElement("Learning Curve Summary");
    //================ END EDIT melville ================
    m_TestsList.setSelectedIndex(0);
    m_TestsButton.setEnabled(true);

    m_PerformBut.setEnabled(true);
    
  }

  
  /**
   * Carries out a t-test using the current configuration.
   */
  protected void performTest() {

    String sigStr = m_SigTex.getText();
    if (sigStr.length() != 0) {
      m_TTester.setSignificanceLevel((new Double(sigStr)).doubleValue());
    } else {
      m_TTester.setSignificanceLevel(0.05);
    }

    String precStr = m_PrecTex.getText();
    if (precStr.length() != 0) {
	m_TTester.setPrecision((new Integer(precStr)).intValue());
    } else {
	m_TTester.setPrecision(2);
    }
    // Carry out the test chosen and biff the results to the output area
    m_TTester.setShowStdDevs(m_ShowStdDevs.isSelected());
    int compareCol = m_CompareCombo.getSelectedIndex();
    int tType = m_TestsList.getSelectedIndex();

    //=============== BEGIN EDIT melville ===============
    String test_selected = (String) m_TestsList.getSelectedValue();
    String name = (new SimpleDateFormat("HH:mm:ss - "))
      .format(new Date())
      + (String) m_CompareCombo.getSelectedItem() + " - "
      + test_selected;
    StringBuffer outBuff = new StringBuffer();
    if(((String) m_CompareCombo.getSelectedItem()).equalsIgnoreCase("%error_reduction"))
	outBuff.append(m_TTester.header("Percentage error reduction"));
    else if(((String) m_CompareCombo.getSelectedItem()).equalsIgnoreCase("top_20%_%error_reduction"))
	outBuff.append(m_TTester.header("Top 20% Percentage error reduction"));
    else outBuff.append(m_TTester.header(compareCol));
    outBuff.append("\n");
    m_History.addResult(name, outBuff);
    m_History.setSingle(name);
    try {
	if (tType < m_TTester.getNumResultsets()) {
	    if(((String) m_CompareCombo.getSelectedItem()).equalsIgnoreCase("%error_reduction")){
		outBuff.append(m_TTester.multiResultsetPercentErrorReduction(tType, m_ErrorCompareCol));
	    }else if(((String) m_CompareCombo.getSelectedItem()).equalsIgnoreCase("top_20%_%error_reduction")){
		outBuff.append(m_TTester.multiResultsetTopNPercentErrorReduction(tType, m_ErrorCompareCol, 0.20));
	    }else outBuff.append(m_TTester.multiResultsetFull(tType, compareCol));
	}
      
      else if (test_selected.equalsIgnoreCase("summary")) {
	  outBuff.append(m_TTester.multiResultsetSummary(compareCol));
      } else if (test_selected.equalsIgnoreCase("ranking")) {
	  outBuff.append(m_TTester.multiResultsetRanking(compareCol));
      }  
      //================ END EDIT melville ================
      
      outBuff.append("\n");
    } catch (Exception ex) {
      outBuff.append(ex.getMessage() + "\n");
    }
    m_History.updateResult(name);
  }

  
  public void setResultKeyFromDialog() {

    ListSelectorDialog jd = new ListSelectorDialog(null, m_ResultKeyList);

    // Open the dialog
    int result = jd.showDialog();
    
    // If accepted, update the ttester
    if (result == ListSelectorDialog.APPROVE_OPTION) {
      int [] selected = m_ResultKeyList.getSelectedIndices();
      String selectedList = "";
      for (int i = 0; i < selected.length; i++) {
	selectedList += "," + (selected[i] + 1);
      }
      Range generatorRange = new Range();
      if (selectedList.length() != 0) {
	try {
	  generatorRange.setRanges(selectedList);
	} catch (Exception ex) {
	  ex.printStackTrace();
	  System.err.println(ex.getMessage());
	}
      }
      m_TTester.setResultsetKeyColumns(generatorRange);
      setTTester();
    }
  }
  
  public void setDatasetKeyFromDialog() {

    ListSelectorDialog jd = new ListSelectorDialog(null, m_DatasetKeyList);

    // Open the dialog
    int result = jd.showDialog();
    //=============== BEGIN EDIT melville ===============
    //Check if learning curves should be generated
    boolean noise = false;
    boolean learning = false;
    boolean fraction = false;
    int learning_key = -1;
    
    // If accepted, update the ttester
    if (result == ListSelectorDialog.APPROVE_OPTION) {
      int [] selected = m_DatasetKeyList.getSelectedIndices();
      String selectedList = "";
      Object [] selected_string = m_DatasetKeyList.getSelectedValues();
      for (int i = 0; i < selected.length; i++) {
	  if(((String)selected_string[i]).toLowerCase().equals("key_noise_levels")){
	      learning_key = i;
	      learning = true;
	      //fraction = true;
	      noise = true;
	  }else	if(((String)selected_string[i]).toLowerCase().equals("key_fraction_instances")){
	      learning_key = i;
	      learning = true;
	      fraction = true;
	  }else if(((String)selected_string[i]).toLowerCase().equals("key_total_instances")  && !learning){ 
	      learning = true;
	      learning_key = i;
	  }else
	      selectedList += "," + (selected[i] + 1);
      }
      
      m_TTester.setLearningCurve(learning);
      m_TTester.setFraction(fraction);
      if(learning){//get points on the learning curve
	  selectedList += "," + (selected[learning_key] + 1);
	  Attribute attr;
	  if(noise){//override fraction
	      attr = m_Instances.attribute("Key_Noise_levels");
	  }else	if(fraction){
	      attr = m_Instances.attribute("Key_Fraction_instances");
	  }else {
	      attr = m_Instances.attribute("Key_Total_instances");
	  }
	  double []pts = new double [attr.numValues()];
	  for(int k=0; k<attr.numValues(); k++){
	      pts[k] = Double.parseDouble(attr.value(k));
	  }
	  Arrays.sort(pts);
	  m_TTester.setPoints(pts);
      }
      //================ END EDIT melville ================

      Range generatorRange = new Range();
      if (selectedList.length() != 0) {
	try {
	  generatorRange.setRanges(selectedList);
	} catch (Exception ex) {
	  ex.printStackTrace();
	  System.err.println(ex.getMessage());
	}
      }
      m_TTester.setDatasetKeyColumns(generatorRange);
      setTTester();
    }
  }

  public void setTestBaseFromDialog() {
    ListSelectorDialog jd = new ListSelectorDialog(null, m_TestsList);

    // Open the dialog
    jd.showDialog();
  }

  /**
   * Save the currently selected result buffer to a file.
   */
  protected void saveBuffer() {
    StringBuffer sb = m_History.getSelectedBuffer();
    if (sb != null) {
      if (m_SaveOut.save(sb)) {
	JOptionPane.showMessageDialog(this,
				      "File saved",
				      "Results",
				      JOptionPane.INFORMATION_MESSAGE);
      }
    } else {
      m_SaveOutBut.setEnabled(false);
    }
  }

  //=============== BEGIN EDIT mbilenko ===============
  /**
   * Plot the currently selected output buffer
   */
  protected void plotOutput() {
    try { 
      StringBuffer sb = m_History.getSelectedBuffer();
      if (sb != null) {
	// dump the output into a temporary file
	File tempDirFile = new File("/tmp");
	final File bufferFile = File.createTempFile("buffer", "", tempDirFile);
	bufferFile.deleteOnExit();

	PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(bufferFile)));
	writer.print(sb);
	writer.close();

	// launch the perl script to process the file
	
	//Process proc = Runtime.getRuntime().exec("perl weka/gui/experiment/plot.pl " + bufferFile.getPath());
	Process proc = Runtime.getRuntime().exec("perl /u/ml/software/weka-latest/weka/gui/experiment/plot.pl " + bufferFile.getPath());
	proc.waitFor();

	// get a list of generated gnuplot scripts
	String[] scriptList = tempDirFile.list(new FilenameFilter() {
	    public boolean accept(File f, String s) { return (s.endsWith(".gplot") && s.startsWith(bufferFile.getName())); }
	  });
	for (int i = 0; i < scriptList.length; i++) {
	  // launch gnuplot
	  scriptList[i] = tempDirFile.getPath() + "/" + scriptList[i];
	  System.out.println(scriptList[i]);
	  proc = Runtime.getRuntime().exec("gnuplot -persist " + scriptList[i]);
	  File plotFile = new File(scriptList[i]);
	  plotFile.deleteOnExit();
	  File dataFile = new File(scriptList[i].replaceAll(".gplot", ".dat"));
	  dataFile.deleteOnExit();
	} 
      } else {
	m_PlotBut.setEnabled(false);
      }
    } catch (Exception e) {
      System.out.println("Problems plotting: " + e);
      e.printStackTrace();
    } 
  } 


  //=============== END EDIT mbilenko ===============
  
  /**
   * Tests out the results panel from the command line.
   *
   * @param args ignored
   */
  public static void main(String [] args) {

    try {
      final JFrame jf = new JFrame("Weka Experiment: Results Analysis");
      jf.getContentPane().setLayout(new BorderLayout());
      final ResultsPanel sp = new ResultsPanel();
      //sp.setBorder(BorderFactory.createTitledBorder("Setup"));
      jf.getContentPane().add(sp, BorderLayout.CENTER);
      jf.addWindowListener(new WindowAdapter() {
	public void windowClosing(WindowEvent e) {
	  jf.dispose();
	  System.exit(0);
	}
      });
      jf.pack();
      jf.setSize(700, 550);
      jf.setVisible(true);
    } catch (Exception ex) {
      ex.printStackTrace();
      System.err.println(ex.getMessage());
    }
  }
}
