import  igraph
import  matplotlib.pyplot as plt


g = igraph.Graph()
# 添加网络中的点
vertex = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
g.add_vertices(vertex)
# 添加网络中的边
edges = [('a', 'c'), ('a', 'e'), ('a', 'b'), ('b', 'd'), ('b', 'g'), ('c', 'e'),
         ('d', 'f'), ('d', 'g'), ('e', 'f'), ('e', 'g'), ('f', 'g')]
g.add_edges(edges)

# 国家名称
g.vs['label'] = ['齐', '楚', '燕', '韩', '赵', '魏', '秦']
# 国家大致相对面积（为方便显示没有采用真实面积）
g.vs['aera'] = [50, 100, 70, 40, 60, 40, 80]


# 点的度
numbers = g.degree()
# 不同国家邻国数量
neighbors = dict(zip(g.vs['label'], numbers))
print (" 度")
print(neighbors)

# 计算中介中心性
betweenness = g.betweenness()
# 保留一位小数
betweenness = [round(i, 1) for i in betweenness]
# 与国家名对应
country_betweenness = dict(zip(g.vs['label'], betweenness))
print('不同国家的中介中心性（枢纽作用）：\n', country_betweenness)

# 计算魏国和齐国的最短路径（如有多条路径，只取其中之一）
path = g.get_shortest_paths('c', 'd')[0]
seq = g.vs.select(path)
print('燕韩之间的最短路径: ', seq['label'])

# 用Igraph内置函数绘图
igraph.plot(g)
plt.show()




















