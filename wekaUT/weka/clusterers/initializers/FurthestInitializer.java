package weka.clusterers.initializers; 

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.clusterers.*;
import weka.core.metrics.*;

public class FurthestInitializer extends Initializer{
    Instances instances;
    Metric metric;
    boolean [] selected;
    public void setClusterer(Clusterer clusterer) throws Exception{
        super.setClusterer(clusterer);
        instances=clusterer.instances;
        metric=clusterer.metric;
		selected=new boolean[instances.numInstances()];
    }
    public Instances initialize() throws Exception{
        Instances centroids=new Instances(instances,0);
        int index;        
		for(int i=0;i<numClusters;i++){
            index=furthest();
			centroids.add(instances.instance(index));
			selected[index]=true;
            System.out.println("FurthestInitializer.initialize centroid "+i+" : "+index);
		}
        return centroids;
    }
    public int getUnselectedIndex(){
		Random random=new Random();
        int index;
		do{
			index=random.nextInt(instances.numInstances());
		}while(selected[index]);
        return index;
    }
    public double distance(Instance instance) throws Exception{
        double nearest=Double.POSITIVE_INFINITY,distance;
        for(int i=0;i<selected.length;i++)if(selected[i]){
            distance=metric.distance(instances.instance(i),instance);
            if(distance<nearest){
                nearest=distance;
            }
        }
        return nearest;
    }
    public int furthest() throws Exception{
        ArrayList indexes=new ArrayList();
        double furthest=Double.NEGATIVE_INFINITY,distance;
        for(int i=0;i<selected.length;i++)if(!selected[i]){
            distance=distance(instances.instance(i));
            if(distance>=furthest){
                furthest=distance;
                indexes.add(new Integer(i));
            }
        }
        Random random=new Random();
        int index=random.nextInt(indexes.size());
        index=((Integer)(indexes.get(index))).intValue();
        return index;
    }
}

