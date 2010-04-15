package weka.clusterers.initializers; 

import java.io.*;
import java.util.*;
import weka.core.*;
import weka.clusterers.*;

public abstract class Initializer implements Cloneable,Serializable,OptionHandler{
  protected Clusterer clusterer=null;
  protected int numClusters;
  public Initializer(){
  }
  public Initializer(Clusterer clusterer) throws Exception{
      setClusterer(clusterer);
  }
  public void setClusterer(Clusterer clusterer) throws Exception{
      this.clusterer=clusterer;
      this.numClusters=clusterer.numberOfClusters();
  }
  public void setNumClusters(int numClusters) throws Exception{
      this.numClusters=clusterer.numberOfClusters();
  }
  public String [] getOptions(){
      String [] options=new String[0];
      return options;
  }
  public void setOptions(String [] options) throws Exception{
  }
  public Enumeration listOptions(){
      Vector vector=new Vector(0);
      return vector.elements();
  }
  public abstract Instances initialize() throws Exception;
}

