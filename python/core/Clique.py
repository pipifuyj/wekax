# Copyright (C) 2010 Chuanren Liu
class Clique(list):
	support=None
	confidence=None
	def __init__(self,list,support=None,confidence=None):
		super(Clique,self).__init__(list)
		self.support=support
		self.confidence=confidence
	@classmethod
	def load(self,line):
		def parse(str):
			if "%"==str[-1]:return float(str[0:-1])/100
			return float(str)
		p,q=line.index("("),line.index(")")
		return self(map(int,line[0:p].split()),*map(parse,line[p+1:q].split()))