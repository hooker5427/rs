import igraph

g = igraph.Graph.Read_GML('./football.gml')

print( g.get_adjlist() )
igraph.plot(g)
print(g.community_label_propagation())

