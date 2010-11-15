# Copyright (C) 2010 Chuanren Liu
from Metric import Metric
class DotP(Metric):
	def __sub__(self,other):
		a=abs(self)
		if a==0:return 1
		b=abs(other)
		if b==0:return 1
		ab=0
		for i in range(len(self)):ab+=self[i]*other[i]
		ab/=a*b
		return 1/(1+ab)