# Copyright (C) 2010 Chuanren Liu
class Clusterer(object):
	K=None
	assignments=None
	distances=None
	centroids=None
	clusters=None
	def __init__(self,K=2):
		self.K=K
	def build(self,instances,centroids):
		numInstances=len(instances)
		self.centroids=centroids
		self.clusters=[None]*self.K
		self.assignments=[None]*numInstances
		self.distances=[None]*numInstances
		i=0
		while True:
			print "Clusterer building loop %d ..."%i
			c=self.cluster(instances)
			self.evaluate(c)
			if c==0:break
			self.centroids=[cluster.mean() for cluster in self.clusters]
			i+=1
	def cluster(self,instances):
		c=0
		for k in range(self.K):self.clusters[k]=instances[0:0]
		for i in range(len(instances)):
			instance=instances[i]
			distances=[instance-centroid for centroid in self.centroids]
			distance=min(distances)
			k=distances.index(distance)
			self.clusters[k].append(instance)
			if k!=self.assignments[i]:
				c+=1
				self.assignments[i]=k
			self.distances[i]=distance
		return c
	def evaluate(self,c=0):
		if c==0:
			print "Converged!"
			print "\n".join(self.assignments)
			return
		print "Moved %d instances"%c
		print "Fitness is %f"%sum(self.distances)
		c=[0]*self.K
		for k in self.assignments:c[k]+=1
		print c