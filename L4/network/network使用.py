import  networkx  as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 生成无向图
edges = np.random.randint(0, 10, size=(  15, 2))



def get_adjmatrix ( data  :dict ) :
    n =  len(data)
    adjlist = np.zeros( (n ,n) , dtype= np.int)
    for k ,   d  in data.items() :
        for  v  in d.keys() :
            adjlist[k][v] = 1
            adjlist[v][k] = 1
    return  adjlist

def check(edges ) :
    edges = np.array( edges)
    minx  =  edges.min()
    maxx = edges.max()
    return  maxx - minx +1 ==  len(set( edges.flatten()))


def duplicate( edges ) :
    g = {}
    for edge in edges:
        s, t = edge[0], edge[1]
        if g.get(s) != None:
            if not t in g[s]:
                g[s].append(t)
            else:
                continue
        else:
            g[s] = [t, ]
    newedges = []
    for k, mylist in g.items():
        for v in mylist:
            newedges.append((k, v))
    return  newedges


while  True :
    edges = duplicate(edges)
    if  check(edges) :
        break
    edges = np.random.randint(0, 10, size=(15, 2))


# 建立一张无向图
g = nx.Graph()
# 添加边长
g.add_edges_from(edges)

position = nx.spring_layout( g )

print ("打印节点")
print (g.nodes)
print ("打印边" )
print( g.edges)

print ("打印邻接矩阵：")
print (get_adjmatrix( g.adj))

print ("d打印各个节点度的信息:")
print (g.degree)













#
#











nx.draw( g ,  position , with_labels= True )


plt.show()



