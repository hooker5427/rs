# coding :utf-8
from datasketch import MinHash, MinHashLSHForest
import re
import jieba
import time

# 1.加载语料库
f = open(r"weibos.txt", 'r', encoding="utf-8")
content = f.read()
# 1. 简单处理
content_list = re.split("[。？！!]", content.replace('\n', '').replace("#", "").strip().rstrip())
content_list = [s.strip().rstrip() for s in content_list if len(content_list) > 0]

# 2.分词

stopwords = [line.encode().decode("utf-8", errors="ignore").strip().rstrip() for line in
             open("stopword.txt", 'r', encoding="utf-8").readlines()]


# 分词函数
def jieba_cut(sentence):
    wordstr = ""
    templist = jieba.cut(sentence)

    for mystr in templist:
        if mystr not in stopwords:
            wordstr += mystr
            wordstr += " "
    return wordstr


# 对item_str创建MinHash
def get_minhash(item_str):
    temp = MinHash()
    for d in item_str:
        temp.update(d.encode('utf8'))
    return temp


# 3.建立分词之后的文档
docment = []
for sentence in content_list:
    item_str = jieba_cut(sentence)
    docment.append(item_str)

# 建立MinHash结构
MinHashList = []
forest = MinHashLSHForest()
for i, line in enumerate(docment):
    hash_codes = get_minhash(line)
    MinHashList.append(hash_codes)
    forest.add(i, hash_codes)

# index所有key，以便可以进行检索
forest.index()

query = '国足输给叙利亚后，里皮坐不住了，直接辞职了'
print("query  str :", query)
# 4. 将item_text进行分词
item_str = jieba_cut(query)
# 得到item_str的MinHash
minhash_query = get_minhash(item_str)

# 5. 查询forest中与m1相似的Top-K个邻居
result = forest.query(minhash_query, 3)
for i in range(len(result)):
    print("vocab_id:", result[i], "jaccard :", minhash_query.jaccard(MinHashList[result[i]]), "text:",
          docment[result[i]].replace(' ', ''))
print("Top 3 邻居", result)
