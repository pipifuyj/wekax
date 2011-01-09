# -*- coding: UTF8 -*-
class Apriori(object):
    globalfreq={}#全局频繁项收集
    maxfreq={}#全局最大频繁项收集
    currfreq={}#遍历过程临时局部频繁项收集
    minsup=None#最小支持度
    transactions=[]#交易记录
    def __init__(self,data,minsup):
        self.transactions=data
        self.minsup=minsup
        self.split=self.transactions[0].split
    def count(self,item):
        if item=='null' or item=='': return  
        if self.currfreq.has_key(item): self.currfreq[item]+=1 
        else: self.currfreq[item]=1
    def remove(self,map):
        for k  in [k for k,v in map.items() if v<self.minsup]:del map[k]
    def comb(self,list1,list2):
        return list(set(list1+list2)) 
    def findFreq(self,currfreq,dep):
        newfreq={}
        C={}#可能频繁项
        keys=currfreq.keys()
        # 根据上次频繁项，生成本次 '可能频繁项' 集合 
        for i in range(len(keys)): 
            for j in range(i+1,len(keys)):
                candidate=self.comb(keys[i].split(self.split),keys[j].split(self.split))
                if not len(candidate) == dep: continue
                candidate.sort()
                candidate=self.split.join(candidate)
                if not candidate in C: C[candidate]=(keys[i],keys[j])
        #  '可能频繁项' 对比 交易数据库  计数
        for k,v in C.items():
            for TID in self.transactions:
                if TID.include(k):
                    if newfreq.has_key(k): newfreq[k]+=1
                    else: newfreq[k]=1
        # 刷选掉 小于 最小支持度 的 频繁项
        self.remove(newfreq)
        for k,v in self.maxfreq.items():
            for k1,v1 in newfreq.items():
                #if self.inTransaction(k,k1):
                if type(self.transactions[0])(k1).include(k):
                    del self.maxfreq[k]
                    break
        # 全局 频繁项 ,最大频繁项  收集 
        for k,v in newfreq.items(): 
            self.globalfreq[k]=v
            self.maxfreq[k]=v
        return newfreq   
    def buildAssociation(self):
        dep=1
        map(self.count,[I for TID in self.transactions for I in TID.toString().split(self.split) ]) 
        self.remove(self.currfreq)
        # 装载全局频繁项 最大频繁项
        for k,v in self.currfreq.items(): 
            self.globalfreq[k]=v
            self.maxfreq[k]=v
        self.observe(dep)
        while self.currfreq:
            dep=dep+1
            self.currfreq = self.findFreq(self.currfreq,dep)
            self.observe(dep)
        # 全局频繁项中去除最大频繁项
        for k,v in self.maxfreq.items():
            if self.globalfreq.has_key(k): del self.globalfreq[k]
        self.evaluate()
    def observe(self,dep):
        print "第"+str(dep)+"次 筛选 频繁项 结束!" 
        print self.currfreq
    def evaluate(self):
        print "===================关联分析评价=================="
        print "频繁项"
        print self.globalfreq
        print "最大频繁项"
        print self.maxfreq
        print "可信度 展现"
        for k,v in  self.globalfreq.items():
            for k1,v1 in self.maxfreq.items():
                if type(self.transactions[0])(k1).include(k):
                    print k,"->",k1,"\t%.1f" %((float(self.maxfreq[k1])/float(self.globalfreq[k]))*100)+"%"
        print "================================================="
