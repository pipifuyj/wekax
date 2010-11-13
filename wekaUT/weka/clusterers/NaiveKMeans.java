package weka.clusterers;

import java.util.*;
import weka.core.*;

public class NaiveKMeans extends Clusterer{
    
	public boolean doEvaluate=false;
	
	public void buildClusterer(Instances data) throws Exception{
		super.buildClusterer(data);
		if(doEvaluate)System.out.print(evaluate());
	}
    public String evaluate() throws Exception{
        String string=super.evaluate();
        if(!doEvaluate)return string;
        for(int i=0,I;i<K;i++){
            string+="Cluster "+i+":";
            I=instanceses[i].numInstances();
            double [] distances=new double[I];
            double [] dis=new double[I];
            int [] indexes=new int[I];
            double distance;
            for(int j=0;j<I;j++){
                distance=metric.distance(centroids.instance(i),instanceses[i].instance(j));
                distances[j]=distance;
                dis[j]=distance;
                indexes[j]=((Integer)(clusters[i].get(j))).intValue();
            }
            Arrays.sort(dis);
            for(int j=0;j<20&&j<I;j++){
                for(int k=0;k<I;k++){
                    if(distances[k]==dis[j]){
                        string+="\tcenter: "+indexes[k];
                        break;
                    }
                }
            }
            for(int j=I-1;j>I-21&&j>-1;j--){
                for(int k=0,kk=I;k<kk;k++){
                    if(distances[k]==dis[j]){
                        string+="\tcircle: "+indexes[k];
                        break;
                    }
                }
            }
        }
        return string;
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
