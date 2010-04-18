package weka.filters;

import java.io.*;
import java.util.*;
import weka.core.*;

public class Split extends Filter{
  public static void filterFile(Filter filter, String [] options) 
    throws Exception {
    Instances mydata = null;
    BufferedReader dat = null;
    BufferedReader lab = null;
    PrintWriter output1 = new PrintWriter(new FileOutputStream("/home/spidr/MyWorkspace/finaldataset/train.arff"));
    PrintWriter output2 = new PrintWriter(new FileOutputStream("/home/spidr/MyWorkspace/finaldataset/test.arff"));
    boolean isSparse=false, isFind = false;
    int id,labval;
String line = "";

    try {
      String data = Utils.getOption('d', options);
      String label = Utils.getOption('l', options);
      String sparse=Utils.getOption('s',options);

      if (data.length() != 0) {
	dat = new BufferedReader(new FileReader(data));
      } else {
	dat = new BufferedReader(new InputStreamReader(System.in));
      }

      if (label.length() != 0) {
	lab = new BufferedReader(new FileReader(label));
      } else {
	lab = new BufferedReader(new InputStreamReader(System.in));
      }

      if(sparse.length()!=0){
          isSparse=true;
      }

      mydata = new Instances(dat);
      mydata.setClassIndex(mydata.numAttributes()-1);

    } catch (Exception ex) {
      String genericOptions = "\nGeneral options:\n\n"
	+ "-h\n"
	+ "\tGet help on available options.\n"
	+ "\t(use -b -h for help on batch mode.)\n"
	+ "-d <file>\n"
	+ "\tThe name of the file containing input instances.\n"
	+ "\tIf not supplied then instances will be read from stdin.\n"
	+ "-l <file>\n"
	+ "\tThe name of the file containing input query label.\n"
	+ "-s <is sparse>\n";
      throw new Exception('\n' + ex.getMessage()
			  +genericOptions);
    }

    Instances train=new Instances(mydata.relationName()+"train",new FastVector(),0);
    Instances test=new Instances(mydata.relationName()+"test",new FastVector(),0);
    Instance instance;

    for(int i = 0; i < mydata.numAttributes(); i ++){
	 train.insertAttributeAt(new Attribute("col"+i),train.numAttributes());
	 test.insertAttributeAt(new Attribute("col"+i),test.numAttributes()); 
    }
output1.println(train.toString());
output2.println(test.toString());

ArrayList qr = new ArrayList();
while ( (line = lab.readLine()) != null) {
		String[] temp = line.split(" ");
		id = Integer.parseInt(temp[0]);
		labval = Integer.parseInt(temp[1]);
		int[] temp1 = new int[2];
		temp1[0] = id-1;
		temp1[1] = labval;
                qr.add(temp1);
}

for(int i = 0; i < mydata.numInstances(); i ++){
	instance = mydata.instance(i);

	isFind = false;
	for(int j = 0; j < qr.size(); j ++){
		int[] temp3 = (int[])(qr.get(j));
		if(temp3[0] == i){
			instance.setValue(mydata.numAttributes()-1,temp3[1]);	
			isFind = true; 
//			qr.remove(j);
			break;
		}
    	}
	if(isSparse){ instance = new SparseInstance(1,instance.toDoubleArray()); }
	else { instance = new Instance(1,instance.toDoubleArray()); }
	if(isFind == false){ output2.println(instance.toString());}
	if(isFind == true) { output1.println(instance.toString());}
}

    if (output1 != null) {
      output1.close();
    }

    if (output2 != null) {
      output2.close();
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







