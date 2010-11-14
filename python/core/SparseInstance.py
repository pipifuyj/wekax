# Copyright (C) 2010 Chuanren Liu
from Instance import Instance
class SparseInstance(Instance):
	index=None
	length=None
	def __init__(self,data=[],index=[],length=0):
		if not index:
			index=[]
			if length:raise Exception()
			length=len(data)
			data=[index.append(i) or data[i] for i in range(length) if data[i]]
		super(SparseInstance,self).__init__(data)
		self.index=index
		self.length=length
	def __getitem__(self,index):
		if index in self.index:return self.data[self.index.index(index)]
		return 0
	def __setitem__(self,index,value):
		if index in self.index:self.data[self.index.index(index)]=value
		else:
			self.index.append(index)
			self.data.append(value)
	def __len__(self):
		return self.length
	def __iter__(self):
		raise NotImplementedError()
	def __unicode__(self):
		return ' '.join(["%d %f"%(self.index[i],self.data[i]) for i in range(len(self.index))])
	@classmethod
	def load(self,line,length=0):
		line=line.split()
		ii=len(line)
		index=[int(line[i])-1 for i in range(0,ii,2)]
		data=[float(line[i]) for i in range(1,ii,2)]
		return self(data,index,length)