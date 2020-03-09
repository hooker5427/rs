import numpy as np
import pandas as pd
import os
from gensim.models import word2vec


# 1. 加载数据
source = './three_kingdoms/segment/cut_text_three_kings.txt'
sentense = word2vec.LineSentence(source)

word2vec.LineSentence(sentense)
# 2.训练
model = word2vec.Word2Vec(sentense,
                          window=3,
                          size=128,
                          iter=5,
                          )
model_save_path_base  ='models'
if not  os.path.exists( model_save_path_base) :
    os.mkdir( model_save_path_base)
model.save('./models/ThreeKing.model')
# print(model.wv.similarity('孙悟空', '猪八戒'))
# print(model.wv.similarity('孙悟空', '孙行者'))
# print(model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))
