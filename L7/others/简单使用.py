# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 12:17
# @Author  : hooker5427

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first docment document.',
    'This document is the second document.',
    'And this  this  is  is the third one.',
    'Is this the first document?',
    ]

# fit + transform
# 一般设置成奇数
N = 3
VEC =CountVectorizer( ngram_range= ( N , N)  )
x = VEC .fit_transform(corpus)

print ("sparse 形式")
print (x)

print(VEC.get_feature_names()) # 词库
print (x.todense())       # sparse  to dense
print ("###################")
# print ("反向解析:\r")
# print(VEC.inverse_transform(x))

print ("打印停用词:")
print (VEC.stop_words)


bags_words  = x.sum(axis = 0 )
#计算词表中每个单词在语料中出现的次数
# 比如 ： "is the" 在句子中一共出现三次
#
words_freq = [(word, bags_words[0, idx]) for word, idx in VEC.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
print (words_freq)