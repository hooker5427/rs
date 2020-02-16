import networkx as nx
import matplotlib.pyplot as plt

G=nx.read_gml('./football.gml')
nx.draw(G,with_labels=True) 
#plt.show() 
print(nx.shortest_path(G, source='Buffalo', target='Kent'))
print(nx.shortest_path(G, source='Buffalo', target='Rice'))



def showdijstra(data) :
    for k ,  path  in data.items( ):
        i =1
        length = len(path)
        for  e in path :
            if i!= length :
                print ( e , "->" ,  end=" ")
            else :
                print (e ,  end=" ")
            i+=1
        print ("\r")


# Dijkstra算法
data = nx.single_source_dijkstra_path(G, 'Buffalo')
showdijstra(data )


# print(nx.multi_source_dijkstra_path(G, {'Buffalo', 'Rice'}))
# # Flody算法
# print(nx.floyd_warshall(G, weight='weight'))
