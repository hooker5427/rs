# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 13:17
# @Author  : hooker5427

import pandas as pd
import numpy as np
import re
import nltk  # pip install nltk

corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'
          ]

labels = ['weather', 'weather', 'animals', 'animals', 'weather', 'animals']

# 第一步：构建DataFrame格式数据
corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus, 'categoray': labels})

# 第二步：构建函数进行分词和停用词的去除
# 载入英文的停用词表
stopwords = nltk.corpus.stopwords.words('english')
# 建立词分割模型
cut_model = nltk.WordPunctTokenizer()


# 定义分词和停用词去除的函数
def Normalize_corpus(doc):
    # 去除字符串中结尾的标点符号
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', string=doc)
    # 是字符串变小写格式
    doc = doc.lower()
    # 去除字符串两边的空格
    doc = doc.strip()
    # 进行分词操作
    tokens = cut_model.tokenize(doc)
    # 使用停止用词表去除停用词
    doc = [token for token in tokens if token not in stopwords]
    # 将去除停用词后的字符串使用' '连接，为了接下来的词袋模型做准备
    doc = ' '.join(doc)

    return doc


# 第三步：向量化函数和调用函数
# 向量化函数,当输入一个列表时，列表里的数将被一个一个输入，最后返回也是一个个列表的输出
Normalize_corpus = np.vectorize(Normalize_corpus)
# 调用函数进行分词和去除停用词
corpus_norm = Normalize_corpus(corpus)

# 第四步：使用TfidVectorizer进行TF-idf词袋模型的构建
from sklearn.feature_extraction.text import TfidfVectorizer

Tf = TfidfVectorizer(use_idf=True)
Tf.fit(corpus_norm)
vocs = Tf.get_feature_names()
corpus_array = Tf.transform(corpus_norm).toarray()
corpus_norm_df = pd.DataFrame(corpus_array, columns=vocs)
print(corpus_norm_df.head())

# 第五步：构建LDA主题模型
from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=2, max_iter=100, random_state=42)
LDA_corpus = np.array(LDA.fit_transform(corpus_array))
LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])
LDA_corpus_one[LDA_corpus[:, 0] < LDA_corpus[:, 1]] = 1
corpus_norm_df['LDA_labels'] = LDA_corpus_one
print(corpus_norm_df.head())


# 第六步：打印每个单词的主题的权重值
tt_matrix = LDA.components_
for tt_m in tt_matrix:
    tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
    tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    # 打印权重值大于0.6的主题词
    tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
    print(tt_dict)
