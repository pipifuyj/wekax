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
			if self.cluster(instances)==0:break
			self.evaluate()
			self.centroids=[cluster.mean() for cluster in self.clusters]
			i+=1
	def cluster(self,instances):
		c=0
		for k in range(self.K):self.clusters[k]=instances[0:0]
		for i in range(len(instances)):
			instance=instances[i]
			distances=[instance-centroid for centroid in self.centroids]
			k=distances.index(min(distances))
			self.clusters[k].append(instance)
			if k!=self.assignments[i]:
				c+=1
				self.assignments[i]=k
		print "Moved %d instances"%c
		return c
	def evaluate(self):
		c=[0]*self.K
		for k in self.assignments:c[k]+=1
		print c