package weka.clusterers.metriclearners; 

import java.util.*;

import weka.core.*;
import weka.core.metrics.*;
import weka.clusterers.MPCKMeans;
import weka.clusterers.InstancePair;

public class FakeGDLearner extends GDMetricLearner {
  public boolean trainMetric(int clusterIdx) throws Exception {
    Init(clusterIdx);
    int violatedConstraints = 0;
    for (int instIdx = 0; instIdx < m_instances.numInstances(); instIdx++) {
      int assignment = m_clusterAssignments[instIdx];
      if (assignment == clusterIdx || clusterIdx == -1) {
	Object list =  m_instanceConstraintMap.get(new Integer(instIdx));
	if (list != null) {
	  ArrayList constraintList = (ArrayList) list;
	  for (int i = 0; i < constraintList.size(); i++) {
	    InstancePair pair = (InstancePair) constraintList.get(i);
	    int firstIdx = pair.first;
	    int secondIdx = pair.second;
	    int otherIdx = (firstIdx == instIdx) ?
	      m_clusterAssignments[secondIdx] : m_clusterAssignments[firstIdx];
	    if (otherIdx != -1) violatedConstraints++;
	  }
	}
      }
    }
    System.out.println(" Cosine: Total constraints violated: " + violatedConstraints/2); 
    return true;
  }

  public String [] getOptions() {

    String [] options = new String [20];
    
    int current = 0;

    options[current++] = "-E";
    options[current++] = "" + m_eta;
    options[current++] = "-D";
    options[current++] = "" + m_etaDecayRate;

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  public void setOptions(String[] options) throws Exception {
    // TODO: add later 
  }

  public Enumeration listOptions() {
    // TODO: add later 
    return null;
  }
}
