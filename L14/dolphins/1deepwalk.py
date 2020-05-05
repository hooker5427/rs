import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

graph=nx.read_gml('../dolphins.gml')

print ("nodes numbers : " , len(graph.nodes ))
print( "edges numbers " ,  len( graph.edges))

print ("all nodes  : " , graph.nodes)

def random_walk( graph ,  node , max_walk_paths ):
    random_walk_list = []
    random_walk_list.append( node)
    for _ in  range(max_walk_paths -1 ):
        temp =  list( graph.neighbors( node))
        temp =  list( set(temp) - set(random_walk_list))
        if len(temp) == 0 :
            break
        node = random.choice( temp)
        random_walk_list.append( node)
    return  random_walk_list

import  pprint
# 游走所有的节点
random_walks = []
for node in  graph.nodes:
    random_walks.append( random_walk( graph , node , 5 ) )


# 使用skip-gram，提取模型学习到的权重
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# 训练skip-gram (word2vec)模型
model = Word2Vec(window = 4, sg = 1, hs = 0,
                 negative = 10, # 负采样
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)
# 从random_walks中创建词汇表
#  更新词表
model.build_vocab(random_walks, progress_per=2)
model.train(random_walks,
            total_examples = model.corpus_count,
            epochs=20, report_delay=1)


# 在二维空间中绘制所选节点的向量
def plot_nodes(word_list):
    # 每个节点的embedding为100维
    X = model[word_list]
    #print(type(X))
    # 将100维向量减少到2维
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plt.figure(figsize=(12,9))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

plot_nodes(model.wv.vocab)
