# Copyright (C) 2010 Chuanren Liu
from random import sample
class Instances(list):
	type=None
	def __init__(self,instances=[],type=None):
		super(Instances,self).__init__(instances);
		self.type=type
	def __getslice__(self,i,j):
		return Instances(super(Instances,self).__getslice__(i,j),self.type)
	def mean(self):
		if self:return sum(self)/len(self)
		return self.type()
	def sample(self,n):
		return Instances(sample(self,n),self.type)
	def load(self,file,*args):
		self+=[self.type.load(line,*args) for line in file]