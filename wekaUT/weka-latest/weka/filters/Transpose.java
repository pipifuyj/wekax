package weka.filters;

import java.io.*;
import java.util.*;
import weka.core.*;

public class Transpose extends Filter{
  public static void filterFile(Filter filter, String [] options) 
    throws Exception {
    Instances data = null;
    Reader input = null;
    PrintWriter output = null;
    boolean isSparse=false;

    try {
      String infileName = Utils.getOption('i', options);
      String outfileName = Utils.getOption('o', options); 
      String classIndex = Utils.getOption('c', options);
      String sparse=Utils.getOption('s',options);
      if (infileName.length() != 0) {
	input = new BufferedReader(new FileReader(infileName));
      } else {
	input = new BufferedReader(new InputStreamReader(System.in));
      }
      if (outfileName.length() != 0) {
	output = new PrintWriter(new FileOutputStream(outfileName));
      } else { 
	output = new PrintWriter(System.out);
      }

      data = new Instances(input, 1);
      if (classIndex.length() != 0) {
	if (classIndex.equals("first")) {
	  data.setClassIndex(0);
	} else if (classIndex.equals("last")) {
	  data.setClassIndex(data.numAttributes() - 1);
	} else {
	  data.setClassIndex(Integer.parseInt(classIndex) - 1);
	}
      }
      
      if(sparse.length()!=0){
          isSparse=true;
      }
    } catch (Exception ex) {
      String genericOptions = "\nGeneral options:\n\n"
	+ "-h\n"
	+ "\tGet help on available options.\n"
	+ "\t(use -b -h for help on batch mode.)\n"
	+ "-i <file>\n"
	+ "\tThe name of the file containing input instances.\n"
	+ "\tIf not supplied then instances will be read from stdin.\n"
	+ "-o <file>\n"
	+ "\tThe name of the file output instances will be written to.\n"
	+ "\tIf not supplied then instances will be written to stdout.\n"
	+ "-c <class index>\n"
	+ "\tThe number of the attribute to use as the class.\n"
	+ "\t\"first\" and \"last\" are also valid entries.\n"
	+ "\tIf not supplied then no class is assigned.\n"
	+ "-s <is sparse>\n";

      throw new Exception('\n' + ex.getMessage()
			  +genericOptions);
    }
    
    Instances instances=new Instances(data.relationName()+"Transposed",new FastVector(),0);
    Instance instance;
    int index=0;
    
    while (data.readInstance(input)) {
        instances.insertAttributeAt(new Attribute("instance"+index),instances.numAttributes());
        index++;
    }
    
    output.println(instances.toString());
    
    for(int i=0,ii=data.numAttributes();i<ii;i++)if(data.attribute(i).isNumeric()){
        if(isSparse){
            instance=new SparseInstance(1,data.attributeToDoubleArray(i));
        }else{
            instance=new Instance(1,data.attributeToDoubleArray(i));
        }
        output.println(instance.toString());
    }
    
    if (output != null) {
      output.close();
    }
  }
  public static void main(String [] argv) {
    try {
	Transpose.filterFile(new Transpose(), argv);
    } catch (Exception ex) {
      System.out.println(ex.getMessage());
    }
  }
}








