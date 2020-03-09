import jieba


# 加载停用词
f = open("./cn_stopwords.txt", encoding="utf-8")
stop_words = f.readlines()
f.close()


# 加载语料数据
path = "./three_kingdoms/source/three_kingdoms.txt"
segments_all = []
with  open(path, 'rb') as f:
    lines = f.readlines()
    for docment in lines:
        segments = []
        docment.decode("utf-8", errors='ignore')
        docment_cut = jieba.cut(docment)
        for word in docment_cut:
            if word not in stop_words:
                segments.append(word)
        segments_all.extend(segments)

print(segments_all)

res_text = ''
for line in segments_all:
    text_one_line = " ".join(line)
    res_text += res_text + '\n'

del segments_all
with open('cut_text_three_kings.txt', "w", newline='', encoding="utf-8", errors="ignore")  as f:
    f.write(res_text)
print("ok!")
