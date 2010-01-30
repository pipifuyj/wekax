package weka.clusterers;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.core.metrics.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class NaiveKMeans extends Clusterer implements OptionHandler{
	private Instances instances;
    private Metric metric=new Euclidean();
	private ReplaceMissingValues replaceMissingValues;
	private int K=2;
	private Instances centroids;
	private int [] assignments;
	
	public void buildClusterer(Instances data) throws Exception{
		replaceMissingValues=new ReplaceMissingValues();
		replaceMissingValues.setInputFormat(data);
		instances=Filter.useFilter(data,replaceMissingValues);
        metric.buildMetric(data);
		centroids=new Instances(instances,K);
		assignments=new int[instances.numInstances()];
		
		Random random=new Random();
		boolean [] selected=new boolean[instances.numInstances()];
		int index;
		for(int i=0;i<K;i++){
			do{
				index=random.nextInt(instances.numInstances());
			}while(selected[index]);
			centroids.add(instances.instance(index));
			selected[index]=true;
		}
		
		boolean done=false;
		while(!done){
			done=true;
			for(int i=0;i<instances.numInstances();i++){
				Instance instance=instances.instance(i);
				int assignment=clusterInstance(instance);
				if(assignment!=assignments[i]){
					done=false;
					assignments[i]=assignment;
				}
			}
			centroids=new Instances(instances,K);
			Instances [] instanceses=new Instances[K];
			for(int i=0;i<K;i++){
				instanceses[i]=new Instances(instances,0);
			}
			for(int i=0;i<instances.numInstances();i++){
				instanceses[assignments[i]].add(instances.instance(i));
			}
			for(int i=0;i<K;i++){
				double [] vals=new double[instances.numAttributes()];
				for(int j=0;j<instances.numAttributes();j++){
					vals[j]=instanceses[i].mean(j);
				}
				centroids.add(new Instance(1.0,vals));
			}
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
	
	public Enumeration listOptions(){
		Vector vector=new Vector(1);
		vector.addElement(new Option("\tnumber of clusters.","N",1,"-N <num>"));
        vector.addElement(new Option("\tmetric.\tdefault=weka.core.metrics.Euclidean","M",1,"-M <metric class>"));
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
	}
	
	public String [] getOptions(){
		String [] options=new String[2];
		int current=0;
		options[current++]="-N";
		options[current++]=Integer.toString(K);
        options[current++]="-M";
        options[current++]=metric.getClass().getName();
		while(current<options.length){
			options[current++]="";
		}
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
