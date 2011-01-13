import math
import sys
from DecisionTree import *
class ID3(DecisionTree):
    """
    This module contains the functions for calculating the information gain of a
    dataset as defined by the ID3 (Information Theoretic) heuristic.
    """
    data=[]
    attributes=[]
    target_attr=None
    tree=None

    def entropy(self,data,target_attr):
        """
        Calculates the entropy of the given data set for the target attribute.
        """
        val_freq = {}
        data_entropy = 0.0

        # Calculate the frequency of each of the values in the target attr
        for record in data:
            if (val_freq.has_key(record[target_attr])):
                val_freq[record[target_attr]] += 1.0
            else:
                val_freq[record[target_attr]] = 1.0

        # Calculate the entropy of the data for the target attribute
        for freq in val_freq.values():
            data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
            
        return data_entropy
        
    def gain(self,data,attr,target_attr):
        """
        Calculates the information gain (reduction in entropy) that would
        result by splitting the data on the chosen attribute (attr).
        """
        val_freq = {}
        subset_entropy = 0.0

        # Calculate the frequency of each of the values in the target attribute
        for record in data:
            if (val_freq.has_key(record[attr])):
                val_freq[record[attr]] += 1.0
            else:
                val_freq[record[attr]] = 1.0

        # Calculate the sum of the entropy for each subset of records weighted
        # by their probability of occuring in the training set.
        for val in val_freq.keys():
            val_prob = val_freq[val] / sum(val_freq.values())
            data_subset = [record for record in data if record[attr] == val]
            subset_entropy += val_prob * self.entropy(data_subset, target_attr)

        # Subtract the entropy of the chosen attribute from the entropy of the
        # whole data set with respect to the target attribute (and return it)
        return (self.entropy(data, target_attr) - subset_entropy)
    
    def load(self,path):
        """
        this function aims to load data from a file line by line
        """
        try:
            file=open(path,"r")
        except IOError:
            print "Error: The file '%s' was not found on this system" % path
            sys.exit(0)
        # Get string line by line from file
        lines = [line.strip() for line in file.readlines()]

        # Remove the attributes from the list of lines and create a list of the attributes.
        lines.reverse()
        self.attributes = [attr.strip() for attr in lines.pop().split(",")] #use list.pop to del and get the last item of this list
        self.target_attr = self.attributes[-1]
        lines.reverse()

        # Create a list of the data in the data file
        self.data = []
        for line in lines:
            self.data.append(dict(zip(self.attributes,[datum.strip() for datum in line.split(",")])))

    def buildDecisionTree(self):
        """
        the main function of ID classifiction
        """
        self.tree=self.create_decision_tree(self.data,self.attributes,self.target_attr,self.gain)
        print self.tree
        print self.classify(self.tree,self.data)
        
    def observe(self):
        print self.tree
