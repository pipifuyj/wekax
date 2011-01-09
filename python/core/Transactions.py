class Transactions(list):
    type=None
    def __init__(self,data=[],type=None):
        super(Transactions,self).__init__(data);
        self.type=type
    def load(self,file):
        self+=[self.type.load(line) for line in file]
