import jieba

# 加载停用词
f = open("./cn_stopwords.txt", encoding="utf-8")
stop_words = f.readlines()
f.close()


# 加载语料
path = "./three_kingdoms/source/three_kingdoms.txt"
f =open( path , 'rb' )
all_text = ''
while True:
    line = f.readline()
    line.decode("utf-8", errors='ignore')
    line_cut = jieba.cut(line)
    temp_list = []
    for word in line_cut:
        if word not in stop_words:
            temp_list.append(word)
    all_text += ' '.join(temp_list)
    del temp_list
    if not line:
        break



# mem killed !
# segments_all = []
# with  open(path, 'rb') as f:
#     lines = f.readlines()
#     for docment in lines :
#         segments = []
#         docment.decode("utf-8" , errors='ignore')
#         docment_cut = jieba.cut(docment)
#         for word in docment_cut:
#             if word not in stop_words:
#                 segments.append(word)
#         segments_all.extend( segments)
#
# print(segments_all)

with open('./three_kingdoms/segment/cut_text_three_kings.txt', "w", newline='', encoding="utf-8", errors="ignore")  as f:
    f.write(all_text)
print("ok!")
