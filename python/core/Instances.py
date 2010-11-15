# Copyright (C) 2010 Chuanren Liu
from random import sample
class Instances(list):
	type=None
	def __init__(self,instances=[],type=None):
		super(Instances,self).__init__(instances);
		self.type=type
	def __getslice__(self,i,j):
		return Instances(super(Instances,self).__getslice__(i,j),self.type)
	def sum(self):
		return self.type.sum(self)
	def mean(self):
		if self:return self.sum()/len(self)
		return self.type()
	def sample(self,n):
		return Instances(sample(self,n),self.type)
	def load(self,file,*args):
		self+=[self.type.load(line,*args) for line in file]