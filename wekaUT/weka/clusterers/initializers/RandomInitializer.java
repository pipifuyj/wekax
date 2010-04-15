package weka.clusterers.initializers; 

import  java.io.*;
import  java.util.*;
import  weka.core.*;
import  weka.clusterers.*;

public class RandomInitializer extends Initializer{
    public Instances initialize() throws Exception{
        Instances instances=clusterer.getInstances();
        Instances centroids=new Instances(instances,0);
		Random random=new Random();
		boolean [] selected=new boolean[instances.numInstances()];
		int index;
		for(int i=0;i<numClusters;i++){
			do{
				index=random.nextInt(instances.numInstances());
			}while(selected[index]);
			centroids.add(instances.instance(index));
			selected[index]=true;
		}
        return centroids;
    }
}

