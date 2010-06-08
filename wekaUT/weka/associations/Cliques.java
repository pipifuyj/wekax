package weka.associations;

import weka.core.FastVector;
import java.io.Reader;
import java.io.IOException;
import java.io.BufferedReader;
import weka.associations.Clique;

public class Cliques extends FastVector{
	public Cliques(Reader reader) throws IOException{
		BufferedReader bufferedReader=new BufferedReader(reader);
		while(bufferedReader.ready()){
			addElement(new Clique(bufferedReader.readLine()));
		}
	}
	public Clique clique(int index){
		return (Clique)elementAt(index);
	}
}