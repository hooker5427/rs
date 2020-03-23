####  Youtube 论文
- https://link.zhihu.com/?target=https%3A//static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf

#### 借鉴 https://zhuanlan.zhihu.com/p/25343518

#### 两阶段推荐系统
+ 召回(Matching)
    - 召回阶段通过itemCF , userCF ,FM 等方式“粗糙”的召回候选商品， 召回阶段视频的数量是百级别了 ，主要考虑快。
+ 排序(Ranking)
    - Ranking阶段对Matching后的视频采用更精细的特征计算user-item之间的排序分，作为最终输出推荐结果的依据。
+ 之所以把推荐系统划分成Matching和Ranking两个阶段，主要是从性能方面考虑的。Matching阶段面临的是百万级视频，单个视频的性能开销必须很小；而Ranking阶段的算法则非常消耗资源，不可能对所有视频都算一遍，快速召回之后， 在进行细粒度的计算。

####  DNN模型设计
- DNN 输入层：
    + 每个视频都会被embedding到固定维度的向量中。用户的观看视频历史则是通过变长的视频序列表达，最终通过加权平均（可根据重要性和时间进行加权）得到固定维度的watch vector作为DNN的输入。用户画像特征：如地理位置，设备，性别，年龄，登录状态等连续或离散特征都被归一化，和watch vector以及search vector做拼接（concatenate）。还有添加特征——样本年龄等
- DNN隐藏层结构
    + 两阶段的设计在隐藏层基本相同。采用的Tower塔式模型，例如第一层1024，第二层512，第三层256，使用ReLU作为激活函数 。

- DNN输出层
    + Training 阶段输出层为softmax层， Serving 阶段直接用user Embedding和video Embedding计算dot-product表示分数，取topk作为候选结果。
    + 最重要问题是在性能, 因此使用类似局部敏感哈希LSH（近似最近邻方法）
- 排序阶段
+ Training最后一层是Weighted LR，Serving时激励函数使用的e^(w*x+b)。
 
