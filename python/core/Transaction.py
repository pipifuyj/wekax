class Transaction(object):
    data=None
    split=","
    def __init__(self,data=[]):
        super(Transaction,self).__init__()
        if isinstance(data,list):
            self.data=data
        if isinstance(data,str):
            self.data=data.split(self.split)
    def __getitem__(self,index):
        return self.data[index]
    def __setitem__(self,index,value):
        self.data[index]=value
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data)
    def __add__(self,other):
        return self.__class__(self.data+other.data)
    def __string__(self):
        return self.split.join(self.data)
    def toString(self):
        return self.split.join(self.data)
    def include(self,other):
        if isinstance(other,str):
            return len(other.split(self.split))==len([i for i in other if i in self.data])
        if isinstance(other,list):
            return len(other)==len([i for i in other if i in self.data])
    @classmethod
    def load(self,line):
        line=line.split(self.split)
        data=[i.replace("\n","") for i in line] 
        return self(data)
