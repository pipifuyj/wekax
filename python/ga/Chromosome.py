# Copyright (C) 2010 Chuanren Liu
class Chromosome(object):
	gene=None
	fitness=None
	def __init__(self,gene=[]):
		self.gene=gene
	def clone(self):
		new=self.__class__(self.gene[:])
		new.fitness=self.fitness
		return new
	def crossover(a,b):
		raise NotImplementedError()
	def __add__(this,that):
		return this.crossover(that)
	def mutation(self):
		raise NotImplementedError()
	def optimize(self):
		raise NotImplementedError()