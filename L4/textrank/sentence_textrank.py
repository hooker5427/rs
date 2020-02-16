# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import jieba.posseg as pseg

file  = open( 'news.txt' , 'r')
text = file.read()
file.close()

sentence  =  text


# 通过TF-IDF获取关键词
keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns') )
# keywords = jieba.analyse.tfidf(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))

for item in keywords:
    print(item[0], item[1])
print('-' * 100)
