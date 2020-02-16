from snownlp import SnowNLP
import operator
import jieba.analyse

file = open('news.txt', 'r')
text = file.read()
file.close()

snow = SnowNLP(text)
# 打印关键词
print(snow.keywords(20))

# TextRank算法
# 生成文章的摘要
newtxt = [(text.find(line), line) for line in snow.summary(10)]
# 排序
newtxt = sorted(newtxt, key=operator.itemgetter(0))

summary = ""
index = 0
for _, line in newtxt:
    if index == len(newtxt) - 1:
        summary += line + "。"
    else:
        summary += line + ","
    index += 1
print(summary)

# 情感分析进行打分   ,返回积极正面的打分

print("文章积极度的得分:\n")
print(snow.sentiments)
# 文本划分得到的句子
print(snow.sentences)

print(" 使用jiaba 分词进行摘要提取")

# 通过TF-IDF获取关键词
keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
# keywords = jieba.analyse.tfidf(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))

for item in keywords:
    print(item[0], item[1])
