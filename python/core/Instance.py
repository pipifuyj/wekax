# Copyright (C) 2010 Chuanren Liu
class Instance(object):
	data=None
	def __init__(self,data=[]):
		super(Instance,self).__init__()
		self.data=data
	def __getitem__(self,index):
		return self.data[index]
	def __setitem__(self,index,value):
		self.data[index]=value
	def __len__(self):
		return len(self.data)
	def __iter__(self):
		return iter(self.data)
	def __add__(self,other):
		return self.__class__([self[i]+other[i] for i in range(len(self))])
	def __radd__(self,*args):
		return self
	def __mul__(self,m):
		return self.__class__([self[i]*m for i in range(len(self))])
	def __div__(self,d):
		return self*(1.0/d)
	@classmethod
	def sum(self,instances):
		if not instances:return self()
		length=len(instances[0])
		data=[0]*length
		for instance in instances:
			i=0
			while i<length:data[i]+=instance[i]
		return self(data)