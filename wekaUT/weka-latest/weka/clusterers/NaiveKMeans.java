package weka.clusterers;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.core.metrics.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.clusterers.initializers.*;

public class NaiveKMeans extends Clusterer implements OptionHandler{
	private Instances instances;
    private Metric metric=new Euclidean();
	private ReplaceMissingValues replaceMissingValues;
	private int K=2;
	private Instances centroids;
    private Instances [] instanceses;
	private int [] assignments;
    private Initializer initializer=new RandomInitializer();
	
	public void buildClusterer(Instances data) throws Exception{
		replaceMissingValues=new ReplaceMissingValues();
		replaceMissingValues.setInputFormat(data);
		instances=Filter.useFilter(data,replaceMissingValues);
        metric.buildMetric(data);
        initializer.setClusterer(this);
        centroids=initializer.initialize();
        instanceses=new Instances[K];
		assignments=new int[instances.numInstances()];
		boolean done=false;
        int loop=0;
		while(!done){
            System.out.println("===***=== "+loop+" ===***===");
			done=true;
			for(int i=0;i<K;i++){
				instanceses[i]=new Instances(instances,0);
			}
			for(int i=0;i<instances.numInstances();i++){
				Instance instance=instances.instance(i);
				int assignment=clusterInstance(instance);
				instanceses[assignment].add(instance);
				if(assignment!=assignments[i]){
					done=false;
					assignments[i]=assignment;
				}
			}
            printClusters();
			centroids=new Instances(instances,K);
			for(int i=0;i<K;i++){
				double [] vals=new double[instances.numAttributes()];
				for(int j=0;j<instances.numAttributes();j++){
					vals[j]=instanceses[i].mean(j);
				}
				centroids.add(new Instance(1.0,vals));
			}
            loop++;
		}
	}
    
    public void printClusters(){
        for(int i=0;i<K;i++){
            System.out.println("Cluster "+i+":");
            System.out.println("\tcentroid: "+centroids.instance(i));
            System.out.println("\tconsists of "+instanceses[i].numInstances()+" instances");
        }
    }
	
	public int clusterInstance(Instance instance) throws Exception{
		double min=Integer.MAX_VALUE;
		int assignment=0;
		for(int i=0;i<K;i++){
			double d=metric.distance(instance,centroids.instance(i));
			if(d<min){
				min=d;
				assignment=i;
			}
		}
		return assignment;
	}
	
	public int numberOfClusters() throws Exception{
		return K;
	}
    
    public Instances getInstances(){
        return instances;
    }
    
    public Metric fetchMetric(){
        return metric;
    }
	
	public Enumeration listOptions(){
		Vector vector=new Vector(3);
		vector.addElement(new Option("\tnumber of clusters.","N",1,"-N <num>"));
        vector.addElement(new Option("\tmetric.\tdefault=weka.core.metrics.Euclidean","M",1,"-M <metric class>"));
        vector.addElement(new Option("\tinitializer.\tdefault=weka.clusters.initializers.RandomInitializer","I",1,"-I <initializer class>"));
		return vector.elements();
	}
	
	public void setOptions(String [] options) throws Exception{
		String string;
		string=Utils.getOption('N',options);
		if(string.length()!=0){
			K=Integer.parseInt(string);
		}
        string=Utils.getOption('M',options);
        if(string.length()!=0){
            metric=(Metric)Utils.forName(Metric.class,string,options);
        }
        string=Utils.getOption('I',options);
        if(string.length()!=0){
            initializer=(Initializer)Utils.forName(Initializer.class,string,options);
        }
	}
	
	public String [] getOptions(){
		String [] options=new String[6];
		int current=0;
		options[current++]="-N";
		options[current++]=Integer.toString(K);
        options[current++]="-M";
        options[current++]=metric.getClass().getName();
        options[current++]="-I";
        options[current++]=initializer.getClass().getName();
		return options;
	}
	
	public static void main(String [] argv){
		try{
			System.out.println(ClusterEvaluation.evaluateClusterer(new NaiveKMeans(),argv));
		}catch(Exception e){
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
	}
}
