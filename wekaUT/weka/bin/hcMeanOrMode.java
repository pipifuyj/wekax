package weka.bin;

import weka.core.OptionHandler;
import java.util.Enumeration;
import java.util.Vector;
import weka.core.Option;
import weka.core.Utils;
import java.io.IOException;
import java.io.FileReader;
import weka.core.Instances;
import weka.associations.Cliques;
import weka.associations.Clique;

public class hcMeanOrMode implements OptionHandler{
	String relation,clique;
	public Enumeration listOptions(){
		Vector vector=new Vector(2);
		vector.addElement(new Option("\trelation(arff) file","",1,"<relation.arff>"));
        vector.addElement(new Option("\tclique file","",1,"<clique.hc>"));
		return vector.elements();
	}
	public void setOptions(String [] options)throws Exception{
		String option;
		option=Utils.getOption(1,options);
		if(option.length()>0)relation=option;
		else throw new Exception("relation not specified");
		option=Utils.getOption(2,options);
		if(option.length()>0)clique=option;
		else throw new Exception("hyperclique not specified");
	}
	public String [] getOptions(){
		String [] options=new String[2];
		int current=0;
		options[current++]=relation;
		options[current++]=clique;
		return options;
	}
	public Instances getInstances()throws IOException{
		Instances dataset=new Instances(new FileReader(relation));
		Cliques cliques=new Cliques(new FileReader(clique));
		Instances result=new Instances(dataset,0);
		for(int i=0,ii=cliques.capacity();i<ii;i++){
			result.add(new Instances(dataset,cliques.clique(i).items).meanOrMode());
		}
		return result;
	}
	public static void main(String [] argv){
		try{
			hcMeanOrMode util=new hcMeanOrMode();
			util.setOptions(argv);
			Instances instances=util.getInstances();
			System.out.print(instances);
		}catch(Exception e){
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
	}
}