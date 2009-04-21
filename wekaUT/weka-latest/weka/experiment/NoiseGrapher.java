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
 *    NoiseGrapher.java
 *    Copyright (C) 2002 Raymond J. Mooney
 *    Modified to read Key of Noise
 */

package weka.experiment;

import java.util.*;
import java.io.*;
import weka.core.*;


/**
 * Class for producing performance graphs for any metric from learning curve results.
 * Currently supports gnuplot format with various types of error bars
 */

public class NoiseGrapher {

    /** Experimental result data in arff format */
    protected Instances data;

    /** Names of datasets in data */
    protected String[] datasets;

    /** Map from scheme + options name to result data in the form of an
	array of Stats's, one for each learning curve point in points */
    protected HashMap schemeMap;

    /** Ordered array of points on learning in number of training examples */
    protected int[] points;

    /** Name of original file of experimental result data in arff format */
    protected String arffFileName;

    /** The name of the performance metric to plot */
    public String metric = "Percent_correct";

    /** Set if desire error bars of particular type */
    public short errorBars = NONE;

    /** errorBar value for no error bars */
    public static short NONE = 0;
    /** errorBar value for error bars using standard deviations */
    public static short STD_DEV = 1;
    /** errorBar value for error bars using 95% confidence intervals */
    public static short CONF_INF = 2;
    /** errorBar value for error bars using min and max values */
    public static short MIN_MAX = 3;

    /** Set if desire error bars based on 95% confidence intervals */
    public boolean confIntErrorBars = false;

    /** The name of the dataset to plot performance for */
    public String dataset;

    /** Create an initial Grapher and load in data, names of datasets,
     * and set of points on learning curve.
     */
    public NoiseGrapher (String arffFileName, short errorBars) throws Exception {
	this.arffFileName = arffFileName;
	this.errorBars = errorBars;
	setData();
	setDatasets();
	setPoints();
	dataset = datasets[0];
    }

    /** Load data for graph in from the given Experiment result file in arff format */
    protected void setData () throws Exception {
	data = new Instances (new BufferedReader(new FileReader(arffFileName)));
    }

    /** Set array of points on learning curve from Key_Noise_levels values in data */
    protected void setPoints() throws Exception {
	Attribute attr = data.attribute("Key_Noise_levels");
	points = new int[attr.numValues()];
	for (int i =0; i < points.length; i++) 
	    points[i] = Integer.parseInt(attr.value(i));
	Arrays.sort(points);
    }    

    /** Set array of points on learning curve from Key_Dataset values in data */
    protected void setDatasets() throws Exception {
	Attribute attr = data.attribute("Key_Dataset");
	datasets = new String[attr.numValues()];
	for (int i =0; i < datasets.length; i++) 
	    datasets[i] = attr.value(i);
	}    


    /** Read in data for the current values of dataset and metric by indexing 
     *	for each scheme+options name an array of Stats objects for each point on the 
     *  learning curve */
    protected void processData () throws Exception {
	schemeMap = new HashMap();
	// Go through each data line in the data
	Enumeration enum = data.enumerateInstances();
	while (enum.hasMoreElements()) {
	    Instance inst = (Instance)enum.nextElement();
	    // If this is not a line for the current dataset, skip it
	    if (!inst.stringValue(data.attribute("Key_Dataset")).equals(dataset))
		continue;
	    // Get the full name of the scheme by concatenating the system
	    // name and the set of system options 
	    String name = inst.stringValue(data.attribute("Key_Scheme")) +
		inst.stringValue(data.attribute("Key_Scheme_options"));
	    // See if this scheme already has and Stats vector for points
	    Stats[] pointsStats = (Stats[])schemeMap.get(name);
	    if (pointsStats == null) {
		// If not create one
		pointsStats = new Stats[points.length];
		schemeMap.put(name, pointsStats);
	    }
	    // Get the number of training instances for this line
	    int point = Integer.parseInt(inst.stringValue(data.attribute("Key_Noise_levels")));
	    // Find the position in the array of points associated with this point
	    int pointPos = Arrays.binarySearch(points, point);
	    // Get the Stats performance metric object for this point
	    Stats stats = pointsStats[pointPos];
	    if (stats == null) {
		// If there is none, create one
		stats = new Stats();
		pointsStats[pointPos] = stats;
	    }
	    Attribute metricAttr = data.attribute(metric);
	    if (metricAttr == null) throw new Error("Unrecognized metric:" + metric);
	    // Get the value of the performance metric for this line
	    double metricValue = inst.value(metricAttr);
	    // Add this value to the Stats object for this scheme and point
	    // that keeps track of the running sum to eventually compute an average
	    stats.add(metricValue);
	}
    }


    /** Generate gnuplot files for plotting a learning curve for the current
     *  dataset and metric.  Assumes a processData was last performed for
     * this case dataset and metric */
    public void gnuplot() throws Exception {
	// Find min and max values of the performance metric
	double yMin=Double.POSITIVE_INFINITY, yMax=Double.NEGATIVE_INFINITY;
	// Iterate though each scheme and each of its plots points
	Iterator schemeEntries = schemeMap.entrySet().iterator();
	// Index of last point on the learning curve (this may differ
	// for different datasets).
	int last_point=-1, last_index=0;
	while (schemeEntries.hasNext()) {
	    Map.Entry schemeEntry = (Map.Entry)schemeEntries.next();
	    Stats[] pointsStats = (Stats[])schemeEntry.getValue();
	    for (int i=0; i < points.length; i++) {
		// First calculate final mean and other summary stats
		
		//PM
		if(pointsStats[i]==null) continue;
		// Keep track of which is the last point on the
		// learning curve on this dataset
		if(points[i]>last_point) {
		    last_point = points[i];
		    last_index = i;
		}
		
		pointsStats[i].calculateDerived();
		
		if (pointsStats[i].mean < yMin)
		    yMin = pointsStats[i].mean;
		if (pointsStats[i].mean > yMax)
		    yMax = pointsStats[i].mean;
	    }
	}
	// Use result file name stem as a stem for plot files
	String fileStem = removeFileExtension(arffFileName);
	// Also include the name of the dataset in the plot-file stem if
	// there is results for more than one dataset in this result file
	if (datasets.length > 1)
	    fileStem = fileStem + dataset;
	String fileName = fileStem + "_" + metric + ".gplot";
	// Create a file for the gnuplot
	PrintWriter out = new PrintWriter(new FileWriter(fileName));
	// Write proper gnuplot commands in this file
	out.println("set xlabel \"Percentage of Noise in Data\"");
	out.println("set ylabel \"" + metric.replace('_', ' ') + "\"");
	out.println("\nset terminal postscript color\nset size 0.75,0.75\n\nset data style linespoints");
	// Move the key of curve names to the lower right corner, good for learning
	// curves and train time plots that go from lower left to top right
	out.println("set key " + 0.85 * points[last_index] + "," +
		    (yMin + 0.25 * (yMax - yMin)));
	out.print("\nplot ");
	// For each scheme, add it to the plot command to plot this scheme's learning curve
	// for the metric and create a data file for the average data for the learning curve points
	schemeEntries = schemeMap.entrySet().iterator();
	while (schemeEntries.hasNext()) {
	    Map.Entry schemeEntry = (Map.Entry)schemeEntries.next();
	    String scheme = cleanSchemeName((String)schemeEntry.getKey());
	    Stats[] pointsStats = (Stats[])schemeEntry.getValue();
	    // Create a data file for this scheme
	    String dataFileName = fileStem + "_" + metric + "_" + scheme;
	    out.print("'" + dataFileName + "' title \""  + scheme + "\"");
	    if (errorBars != NONE) 
		out.print(", '" + dataFileName + "' notitle with errorbars");
	    if (schemeEntries.hasNext())
		out.print(", ");
	    PrintWriter dataOut = new PrintWriter(new FileWriter(dataFileName));
	    // Write out a line for each data point on the learning curve for the metric
	    for (int i=0; i <= last_index; i++) {
		dataOut.print(points[i] + " " + pointsStats[i].mean);
		// Add a third (and maybe fourth) entry for the error bar.
		// Just a third indicates a delta about the mean, a third
		// and fourth indicates a lower and upper bound
		if (errorBars == STD_DEV) {
		    dataOut.print(" " + pointsStats[i].stdDev);
		}
		else if (errorBars == CONF_INF) {
		    // a 95% confidence interval is a delta of 1.96 standard deviations
		    dataOut.print(" " + 1.96 * pointsStats[i].stdDev);
		}
		else if (errorBars == MIN_MAX) {
		    dataOut.print(" " + pointsStats[i].min + " " + pointsStats[i].max);
		}
		dataOut.println("");
		
	    }
	    dataOut.close();
	}
	out.close();
    }

    /** Clean the name of a scheme to make it appropriate for a file name */
    private String cleanSchemeName(String schemeName) {
	return Utils.removeSubstring(schemeName, "weka.classifiers.").replace(' ','_');
    }
    
    /** Return the name of a file with the extension removed */
    public static String removeFileExtension(String fileName) {
	int pos = fileName.lastIndexOf(".");
	if (pos == -1)
	    return fileName;
	else
	    return fileName.substring(0,pos);
    }

    /** Produce a gnuplot for each dataset in the result file */
    public void gnuplotAllDatasets () throws Exception{
	for(int i =0; i < datasets.length; i++) {
	    dataset = datasets[i];
	    processData();
	    gnuplot();	
	}
    }    

    /** Create gnuplot graphs of  learning curves. The first argument should
     * be the name of an arff file of experimental result for a learning curve experiment.
     * If present, the second argument should be the name of a performance metric in
     * result file to plot (which defaults to Percent_correct). Options are:
     * <ul>
     * <li> -s: Plot error bars of standard deviations.
     * <li> -c: Plot error bars of 95% confidence intervals.
     * <li> -m: Plot error bars of min and max values.
     *</ul>
     */
    public static void main (String[] args) throws Exception {
	int current = 0;
	short errorBars = NONE;
	if (args[current].equals("-s")){
	    errorBars = STD_DEV;
	    current++;
	}
	else if (args[current].equals("-c")){
	    errorBars = CONF_INF;
	    current++;
	}
	else if (args[current].equals("-m")){
	    errorBars = MIN_MAX;
	    current++;
	}
	NoiseGrapher noisegrapher = new NoiseGrapher(args[current++],errorBars);
	if (args.length > current)
	    noisegrapher.metric = args[current++];
	noisegrapher.gnuplotAllDatasets();
    }
}



