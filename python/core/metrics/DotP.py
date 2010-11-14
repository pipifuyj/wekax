# Copyright (C) 2010 Chuanren Liu
from math import sqrt
class DotP(object):
	def __sub__(self,other):
		aa=bb=ab=0
		for i in range(len(self)):
			aa+=self[i]**2
			bb+=other[i]**2
			ab+=self[i]*other[i];
		if aa==0 or bb==0:return 1
		ab/=sqrt(aa)*sqrt(bb)
		return 1/(1+ab)