#### Thinking1   什么是Graph Embedding，都有哪些算法模型?
+ Graph Embedding 是一种利用图这种数据结构对数据进行特征表示的形式。 
+ 图表征的算法有：
    - deep walk
        - step1 : 用户商品行为序列进行构图
        - step2 ：截断随机游走生成带推荐的序列候选集
        - step3 : 推荐用户还没有进行交互的商品
    - node2vec 
        - 改进deep walk ，通过p,q 两个超参数调节算法特性，调整算法是关注局部特征还是宏观特征。
        - step1 ：建图
        - step2 : 建立node2vec模型
        - step3 : p,q 进行表格搜索，可视化观察效果（降维处理）     

#### Thinking2   如何使用Graph Embedding在推荐系统，比如NetFlix 电影推荐，请说明简要的思路?
+ NetFlix数据集搜集< user, item , rating >的序列对。
    + 第一步是生成物品关系图，通过用户行为序列可以生成物品相关图，利用相同属性、相同类别等信息，也可以通过这些相似性建立物品之间的边，从而生成基于内容的knowledge graph。
    + 基于knowledge graph生成的物品向量可以被称为补充信息（side information）embedding向量。

#### Thinking3   在交通流量预测中，如何使用Graph Embedding，请说明简要的思路 ?
+ 通过路网信息建立图的邻接矩阵， 交通流量作为权重，使用 GCN生成embeddding 

#### Thinking4：  在文本分类中，如何使用Graph Embedding，请说明简要的思路?
+ 图中节点分类两类:文档节点，单词节点。 文档-单词的边权重采用tf-idf算法生成， 防疫文档和单词之间的关系；单词与单词之间边的权重采用基于共线矩阵的方法进行相似度的计算，可以采用word2vec基于滑动窗口的算法进行计算单词与单词之间的关系。 
+ 建图后，通过GCN网络进行学习，完成分类任务。

