# Copyright (C) 2010 Chuanren Liu
from Instance import Instance
class SparseInstance(Instance):
	index=None
	length=None
	dict=None
	def __init__(self,data=[],index=[],length=0):
		if not index:
			index=[]
			if length:raise Exception()
			length=len(data)
			data=[index.append(i) or data[i] for i in range(length) if data[i]]
		super(SparseInstance,self).__init__(data)
		self.index=index
		self.length=length
		self.dict={}
	def __index__(self,index):
		if index in self.dict:return self.dict[index]
		self.dict[index]=self.index.index(index)
		return self.dict[index]
	def __getitem__(self,index):
		if index in self.index:return self.data[self.__index__(index)]
		return 0
	def __setitem__(self,index,value):
		if index in self.index:self.data[self.__index__(index)]=value
		else:
			self.index.append(index)
			self.data.append(value)
	def __len__(self):
		return self.length
	def __iter__(self):
		raise NotImplementedError()
	def __add__(self,other):
		index=self.index[:]
		data=self.data[:]
		otherIndex=set(other.index)
		intersection=otherIndex.intersection(index)
		for i in intersection:data[self.__index__(i)]+=other.data[other.__index__(i)]
		otherIndex=otherIndex.difference(intersection)
		index+=list(otherIndex)
		data+=[other.data[other.__index__(i)] for i in otherIndex]
		return self.__class__(data,index,self.length)
	def __mul__(self,m):
		return self.__class__([d*m for d in self.data],self.index[:],self.length)
	def __unicode__(self):
		return ' '.join(["%d %f"%(self.index[i],self.data[i]) for i in range(len(self.index))])
	@classmethod
	def sum(self,instances):
		if not instances:return self()
		data=[0]*len(instances[0])
		for instance in instances:
			for i in range(len(instance.index)):data[instance.index[i]]+=instance.data[i]
		return self(data)
	@classmethod
	def load(self,line,length=0):
		line=line.split()
		ii=len(line)
		index=[int(line[i])-1 for i in range(0,ii,2)]
		data=[float(line[i]) for i in range(1,ii,2)]
		return self(data,index,length)