__all__=[]
from os import listdir
from os.path import dirname
for file in listdir(dirname(__file__)):
    if file[-3:]=='.py':
        file=file[0:-3]
        if file!='__init__':
            __all__.append(file)
for key in locals().keys():
    if key[0:2]!='__' or key[-2:]!='__':
        exec "del %s"%key
del key
for module in __all__:
    exec "from %s import %s"%(module,module)
if 'module' in locals():del module