package weka.associations;

import java.util.regex.Pattern;
import java.util.regex.Matcher;
import weka.core.Utils;

public class Clique{
	public int [] items;
	public double support=0;
	public double confidence=0;
	public Clique(String line){
		Pattern pattern=Pattern.compile("^(.+)\\((.+)\\s+(.+)\\)");
		Matcher matcher=pattern.matcher(line);
		if(matcher.find()){
			items=Utils.toInt(matcher.group(1).split("\\s+"));
			support=Double.valueOf(matcher.group(2)).doubleValue();
			confidence=Double.valueOf(matcher.group(3)).doubleValue();
		}
	}
}