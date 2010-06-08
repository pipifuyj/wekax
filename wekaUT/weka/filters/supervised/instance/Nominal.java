package weka.filters.supervised.instance;

import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.core.OptionHandler;
import java.util.Enumeration;
import java.util.Vector;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Utils;

public class Nominal extends Filter implements SupervisedFilter,OptionHandler{
	public String [] getOptions(){
		String [] options=new String[0];
		return options;
	}
	public void setOptions(String [] options){
	}
	public Enumeration listOptions(){
		Vector vector=new Vector(0);
		return vector.elements();
	}
	public boolean batchFinished()throws Exception{
		Instances input=getInputFormat();
		String relation=input.relationName();
		Instances output=new Instances(relation);
		int numAttributes=input.numAttributes();
		int numInstances=input.numInstances();
		for(int i=0;i<numAttributes;i++){
			FastVector vector=new FastVector();
			for(int j=0;j<numInstances;j++){
				double value=input.instance(j).value(i);
				String string=String.valueOf(value);
				if(vector.indexOf(string)==-1)vector.addElement(string);
			}
			Attribute attribute=new Attribute(input.attribute(i).name(),vector);
			output.appendAttribute(attribute);
		}
		setOutputFormat(output);
		for(int i=0;i<numInstances;i++)push(input.instance(i));
		return super.batchFinished();
	}
	public static void main(String [] argv){
		try{
			if(Utils.getFlag('b',argv))Filter.batchFilterFile(new Nominal(),argv);
		    else Filter.filterFile(new Nominal(),argv);
		}catch(Exception ex){
			System.out.println(ex.getMessage());
		}
	}
}