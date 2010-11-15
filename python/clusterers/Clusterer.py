# Copyright (C) 2010 Chuanren Liu
class Clusterer(object):
	K=None
	assignments=None
	centroids=None
	clusters=None
	def __init__(self,K=2):
		self.K=K
	def build(self,instances,centroids):
		self.centroids=centroids
		self.clusters=[None]*self.K
		self.assignments=[None]*len(instances)
		i=0
		while True:
			print "Clusterer building loop %d ..."%i
			if self.cluster(instances):break
			self.centroids=[cluster.mean() for cluster in self.clusters]
			i+=1
	def cluster(self,instances):
		done=True
		for k in range(self.K):self.clusters[k]=instances[0:0]
		for i in range(len(instances)):
			instance=instances[i]
			distances=[instance-centroid for centroid in self.centroids]
			k=distances.index(min(distances))
			self.clusters[k].append(instance)
			if k!=self.assignments[i]:
				print "Assign instance %d to cluster %d"%(i,k)
				done=False
				self.assignments[i]=k
		return done
	def evaluate(self,classes):
		result=[]
		M=[None]*self.K
		for k in range(self.K):M[k]=[0]*self.K
		for i in range(len(classes)):M[self.assignments[i]][classes[i]]+=1
		for i in range(self.K):result.append("\t".join(map(str,M[i])))
		return "\n".join(result)