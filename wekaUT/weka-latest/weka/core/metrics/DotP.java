package weka.core.metrics;

import java.util.ArrayList;
import java.io.Serializable;
import weka.core.*;

public class DotP extends Metric{  
   public void buildMetric(int numAttributes) throws Exception{
       m_numAttributes=numAttributes;
       m_attrIdxs=new int[numAttributes];
       for(int i = 0;i< numAttributes;i++){
           m_attrIdxs[i]=i;
       }
   }
   
   public void buildMetric(int numAttributes, String[] options) throws Exception{
       buildMetric(numAttributes);
   }
   
   public void buildMetric(Instances data) throws Exception{
       m_numAttributes=data.numAttributes();
       m_classIndex=data.classIndex();
       if(m_classIndex!=-1)m_numAttributes--;
       System.out.println("About to build metric with " +m_numAttributes+ " attributes");
       buildMetric(m_numAttributes);
   }
  
   public double distance(Instance instance1,Instance instance2) throws Exception{
       return distanceNonWeighted(instance1,instance2);
   }
   
   public double similarity(Instance instance1,Instance instance2) throws Exception{
       return similarityNonWeighted(instance1,instance2);
   }
   
   public double penalty(Instance instance1,Instance instance2) throws Exception{
       double similarity=similarity(instance1,instance2);
       return 1-similarity;
   }
   
   public double penaltySymmetric(Instance instance1,Instance instance2) throws Exception{
       return penalty(instance1,instance2);
   }
   
   public double distanceNonWeighted(Instance instance1,Instance instance2) throws Exception{
       return 1/(1+similarityNonWeighted(instance1,instance2));
   }
   
   public double similarityNonWeighted(Instance instance1,Instance instance2) throws Exception{
       double similarity=0;
       double [] values1=instance1.toDoubleArray();
       double [] values2=instance2.toDoubleArray();
       double length1=0,length2=0;
       for (int i=0;i<values1.length;i++){
           if(i==m_classIndex)continue;
           similarity+=values1[i]*values2[i];
           length1+=values1[i]*values1[i];
           length2+=values2[i]*values2[i];
       }
       if(length1!=0&&length2!=0){
           similarity/=Math.sqrt(length1)*Math.sqrt(length2);
       }
       return similarity;
   }
   
   public boolean isDistanceBased(){
       return false;
   }
}
