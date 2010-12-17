# Copyright (C) 2010 Chuanren Liu
from random import random,shuffle
class Organism(object):
	chromosome=None
	fitness=None
	best=None
	worst=None
	def __init__(self,chromosome=[]):
		self.chromosome=chromosome
	def selection(self):
		l,s=len(self.chromosome),sum(self.fitness)
		if not s:return
		p=l/s
		proportion=map(lambda x:int(round(x*p)),self.fitness)
		s=sum(proportion)
		if s<l:proportion[proportion.index(max(proportion))]+=l-s
		elif s>l:
			s-=l
			for p in sorted(proportion):
				if p==0:continue
				if p<s:
					s-=p
					proportion[proportion.index(p)]=0
				else:
					proportion[proportion.index(p)]-=s
					break
		for i in range(l):self.chromosome+=[self.chromosome[i]]*proportion[i]
		self.chromosome=self.chromosome[l:]
	def crossover(self,probability):
		l=len(self.chromosome)
		shuffle(self.chromosome)
		for i in range(1,l,2):
			if random()<probability:
				self.chromosome[i-1:i+1]=self.chromosome[i-1]+self.chromosome[i]
	def mutation(self,probability):
		for i in range(len(self.chromosome)):
			if random()<probability:
				self.chromosome[i].mutation()
	def elitism(self,chromosome):
		self.chromosome[self.worst]=chromosome
		self.fitness[self.worst]=chromosome.fitness
	def optimize(self):
		for chromosome in self.chromosome:chromosome.optimize()
		self.fitness=[chromosome.fitness for chromosome in self.chromosome]
		self.best=self.fitness.index(max(self.fitness))
		self.worst=self.fitness.index(min(self.fitness))
	def generate(self):
		self.selection()
		self.crossover()
		self.mutation()
		self.optimize()
		self.elitism()