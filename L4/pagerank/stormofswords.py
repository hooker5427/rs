import networkx as  nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  operator
import  warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("./stormofswords.csv")
edges_weights  = []
for  lineid , line in data.iterrows():
    source  , target , weight  = line.Source ,line.Target ,line.Weight
    edges_weights.append ( (source  , target , weight ) )


def show_graph(graph, type = 'spring_layout'):
    if type == 'spring_layout':
        positions=nx.spring_layout(graph)
    if type == 'circular_layout':
        positions=nx.circular_layout(graph)

    nodesize = [x['pagerank']*20000 for v,x in graph.nodes(data=True)]
    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.4)
    nx.draw_networkx_edges(graph, positions, edge_size=edgesize, alpha=0.2)
    nx.draw_networkx_labels(graph, positions, font_size=12)
    plt.show()


G = nx.DiGraph()
G.add_weighted_edges_from(edges_weights )
# 计算每个节点（人）的PR值，并作为节点的pagerank属性
pagerank = nx.pagerank(G , alpha =1  )
# 获取每个节点的pagerank数值
pagerank_list = {node: rank for node, rank in pagerank.items()}
# 将pagerank数值作为节点的属性
nx.set_node_attributes(G, name = 'pagerank', values=pagerank_list)
# 展示全部
show_graph(G)
threshold =  0.6

pagerank_list = sorted(pagerank_list.items(), key= operator.itemgetter(1) ,reverse= True )


small_G  = G.copy()
sum = 0
remove_nodes =[]
for node_index, info in enumerate(pagerank_list):
    node, rank = info
    sum += rank
    if sum > threshold:
        remove_nodes.append(node)
small_G.remove_nodes_from(remove_nodes)
show_graph(small_G)



small_G = G.copy()
threshold = 0.008
for info in pagerank_list:
    node  , pr  = info
    if pr < threshold :
        small_G.remove_node(node)
show_graph(small_G)



