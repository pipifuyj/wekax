# Copyright (C) 2010 Chuanren Liu
from DotP import DotP
class SparseDotP(DotP):
	def __sub__(self,other):
		a=abs(self)
		if a==0:return 1
		b=abs(other)
		if b==0:return 1
		ab=0
		index=set(self.index).intersection(other.index)
		for i in index:ab+=self.data[self.__index__(i)]*other.data[other.__index__(i)]
		ab/=a*b
		return 1/(1+ab)
		