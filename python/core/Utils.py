# Copyright (C) 2010 Chuanren Liu
from math import log
class Utils(object):
	@classmethod
	def matrix(self,m,n):
		M=[None]*m
		for i in range(m):M[i]=[0]*n
		return M
	@classmethod
	def evaluate(self,assignments,classes):
		N=min(len(assignments),len(classes))
		ms,ns=set(assignments),set(classes)
		m,n=max(ms)+1,max(ns)+1
		M=self.matrix(m,n)
		for i in range(N):M[assignments[i]][classes[i]]+=1
		M1=[sum(M[i]) for i in range(m)]
		M2=[sum([M[i][j] for i in range(m)]) for j in range(n)]
		p1=[float(i)/N for i in M1]
		p2=[float(j)/N for j in M2]
		p=[[M1[i] and M[i][j]/float(M1[i]) for j in range(n)] for i in range(m)]
		r=[[M2[j] and M[i][j]/float(M2[j]) for j in range(n)] for i in range(m)]
		f=[[M[i][j] and 2*p[i][j]*r[i][j]/(p[i][j]+r[i][j]) for j in range(n)] for i in range(m)]
		Es=[sum([-p[i][j]*log(p[i][j],2) for j in range(n) if p[i][j]]) for i in range(m)]
		Ps=[max(p[i]) for i in range(m)]
		Fs=[max([f[i][j] for i in range(m)]) for j in range(n)]
		E=sum([p1[i]*Es[i] for i in range(m)])
		P=sum([p1[i]*Ps[i] for i in range(m)])
		F=sum([p2[j]*Fs[j] for j in range(n)])
		H1=sum([-p1[i]*log(p1[i],2) for i in range(m) if p1[i]])
		H2=sum([-p2[j]*log(p2[j],2) for j in range(n) if p2[j]])
		NMI=(H2-E)/(H1*H2)**0.5
		summary="E=%f\t P=%f\t F=%f\t NMI=%f"%(E,P,F,NMI)
		return locals()