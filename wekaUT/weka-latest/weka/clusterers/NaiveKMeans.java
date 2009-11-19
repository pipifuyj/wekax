package weka.clusterers;

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class NaiveKMeans extends Clusterer implements OptionHandler{
	private Instances instances;
	private ReplaceMissingValues replaceMissingValues;
	private int K;
	private Instances centroids;
	private int [] assignments;
	
	public void buildClusterer(Instances data) throws Exception{
		replaceMissingValues=new ReplaceMissingValues();
		replaceMissingValues.setInputFormat(data);
		instances=Filter.useFilter(data,replaceMissingValues);
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
			double d=distance(instance,centroids.instance(i));
			if(d<min){
				min=d;
				assignment=i;
			}
		}
		return assignment;
	}
	
	private double distance(Instance a,Instance b){
		double aa=0,bb=0,ab=0;
		for(int i=0;i<instances.numAttributes();i++){
			if(i==instances.classIndex())continue;
			aa+=a.value(i)*a.value(i);
			bb+=b.value(i)*b.value(i);
			ab+=a.value(i)*b.value(i);
		}
		double cos=ab/Math.sqrt(aa*bb);
		double acos=Math.acos(cos);
		return acos;
	}
	
	public int numberOfClusters() throws Exception{
		return K;
	}
	
	public Enumeration listOptions(){
		Vector vector=new Vector(1);
		vector.addElement(new Option("\tnumber of clusters.","N",1,"-N <num>"));
		return vector.elements();
	}
	
	public void setOptions(String [] options) throws Exception{
		String string;
		string=Utils.getOption('N',options);
		if(string.length()!=0){
			K=Integer.parseInt(string);
		}
	}
	
	public String [] getOptions(){
		String [] options=new String[2];
		int current=0;
		options[current++]="-N";
		options[current++]=Integer.toString(K);
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
