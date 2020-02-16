import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt


# 数据加载
G=nx.read_gml('./dolphins.gml')
# G = nx.read_gml('./football.gml')
# G = nx.read_gml('./karate.gml')
# print (G.edges) 打印边信息
# with_label 参数必须参考.gml 文件 是否有这样的信息  否者绘图报错


nx.draw(G  ,with_labels= True )
plt.show()

'''
import csv
def save_to_csv(data) :
    out_f = open("./karate.csv",'w',newline='')
    writer = csv.writer(out_f)
    i =0
    for line in data :
        if i== 0 :
            writer.writerow( ["source" , "target"])
        else :
            writer.writerow( line )
        i+=1
    out_f.close()

save_to_csv(G.edges)

'''


# 社区发现 ， 针对无向图
communities = list(community.label_propagation_communities(G))
print(communities)
print(len(communities))
