package weka.core;

import java.util.*;

public class Dict extends HashMap{
	public Dict(){
		super();
	}
	public Dict(String [] options){
		super();
		String key=null,value;
		for(int i=0;i<options.length;i++){
			if(options[i].startsWith("---"))break;
			if(key==null){
				if(options[i].startsWith("--")){
					key=options[i].substring(2);
					options[i]="";
				}
				else if(options[i].startsWith("-")){
					key=options[i].substring(1);
					options[i]="";
				}
			}else{
				if(options[i].startsWith("--")){
					put(key,true);
					key=options[i].substring(2);
				}else if(options[i].startsWith("-")){
					put(key,true);
					key=options[i].substring(1);
				}else{
					put(key,options[i]);
					key=null;
				}
				options[i]="";
			}
		}
		if(key!=null)put(key,true);
	}
	public String getString(Object key){
		return (String)get(key);
	}
	public Integer getInteger(Object key){
		try{
			return (Integer)get(key);
		}catch(Exception e){
			return Integer.valueOf(getString(key));
		}
	}
	public int getInt(Object key){
		return getInteger(key).intValue();
	}
	public Boolean getBoolean(Object key){
		try{
			return (Boolean)get(key);
		}catch(Exception e){
			return Boolean.valueOf(getString(key));
		}
	}
	public boolean getBool(Object key){
		return getBoolean(key).booleanValue();
	}
	public Object put(Object key,int value){
		Object old=get(key);
		put(key,new Integer(value));
		return old;
	}
	public Object put(Object key,boolean value){
		Object old=get(key);
		put(key,new Boolean(value));
		return old;
	}
}
