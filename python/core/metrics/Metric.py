# Copyright (C) 2010 Chuanren Liu
class Metric(object):
	def __abs__(self):
		if hasattr(self,'abs'):return self.abs
		self.abs=0
		for d in self.data:self.abs+=d**2
		self.abs=self.abs**0.5
		return self.abs
	def __sub__(self,other):
		raise NotImplementedError()