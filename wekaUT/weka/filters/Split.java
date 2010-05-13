package weka.filters;

import java.io.*;
import java.util.*;
import weka.core.*;

public class Split extends Filter{
  public static void filterFile(Filter filter, String [] options) 
    throws Exception {
    Instances instances = null;
    Instances ins = null;
    BufferedReader reader = null;
    Dict opt=new Dict();
    try {
        opt.put("class", Utils.getInt('c',options));
        opt.put("data", Utils.getOption('d',options));
        opt.put("line", Utils.getInt('l',options));
    } catch (Exception ex) {
      String genericOptions = "\nGeneral options:\n\n"
	+ "-d <file>\n"
	+ "\tThe name of the file containing input instances.\n"
	+ "\tIf not supplied then instances will be read from stdin.\n"
	+ "-s <is sparse>\n";
      throw new Exception(ex.getMessage()+genericOptions);
    }
    if(opt.getString("data").length()!=0){
  	  reader=new BufferedReader(new FileReader(opt.getString("data")));
    }else{
  	  reader=new BufferedReader(new InputStreamReader(System.in));
    }
    instances = new Instances(reader);
    int p=instances.numInstances()/opt.getInt("line");
    if(opt.getInt("class")<0)opt.put("class",instances.numAttributes()-1);
    instances.setClassIndex(opt.getInt("class"));
    Attribute classAttribute=instances.classAttribute();
    int numClasses=classAttribute.numValues();
    int [] count=new int[numClasses];
    int [][][] keys=new int[numClasses][][];
    for(int i=0;i<numClasses;i++){
    	count[i]=instances.numInstancesWithClass(i)/p;
    	keys[i]=Utils.chunk(instances.indicesWithClass(i),count[i]);
    }
    Attribute indexAttribute=new Attribute("index");
    instances.insertAttributeAt(indexAttribute,0);
    for(int i=0,ii=instances.numInstances();i<ii;i++)instances.instance(i).setValue(indexAttribute,i);
    String path=opt.getString("data");
    path=path.substring(0,path.lastIndexOf("."));
    for(int i=0;i<p;i++){
    	ins=new Instances(instances,0);
    	for(int j=0;j<numClasses;j++){
    		for(int k=0,kk=keys[j][i].length;k<kk;k++){
    			
    			ins.add(instances.instance(keys[j][i][k]));
    		}
    	}
    	PrintWriter writer=new PrintWriter(new FileOutputStream(path+"-"+i+".arff"));
    	writer.print(ins.toString());
    }
  }
  public static void main(String [] argv) {
    try {
	Split.filterFile(new Split(), argv);
    } catch (Exception ex) {
      System.out.println(ex.getMessage());
    }
  }
}
