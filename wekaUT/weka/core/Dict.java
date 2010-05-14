package weka.core;

import java.util.*;

public class Dict extends HashMap{
	public String getString(Object key){
		return (String)get(key);
	}
	public Integer getInteger(Object key){
		return (Integer)get(key);
	}
	public int getInt(Object key){
		try{
			return getInteger(key).intValue();
		}catch(Exception e){
			return Integer.parseInt(getString(key));
		}
	}
	public Boolean getBoolean(Object key){
		return (Boolean)get(key);
	}
	public boolean getBool(Object key){
		try{
			return getBoolean(key).booleanValue();
		}catch(Exception e){
			return Boolean.valueOf(getString(key));
		}
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