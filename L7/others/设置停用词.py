# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 13:00
# @Author  : hooker5427

#从文件导入停用词表
stpwrdpath ="C:\\Users\\Administrator\\Desktop\\中文停用词库.txt"
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('utf-8')  # 提用词提取
#将停用词表转换为list
stpwrdlst = stopword.splitlines()
# 从sklearn.feature_extraction.text里导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# 对CountVectorizer进行初始化（去除中文停用词）
count_vec=CountVectorizer(stop_words=stpwrdlst) #创建词袋数据结构
X_count_train = count_vec.fit_transform(all_list[:2])  #<class 'scipy.sparse.csr.csr_matrix'>
# 将原始训练和测试文本转化为特征向量
X_count_train= X_count_train.toarray()
X_count_test = count_vec.transform(all_list[2]).toarray()
print(X_count_train)
#词汇表
print('\nvocabulary list:\n\n',count_vec.get_feature_names())
print( '\nvocabulary dic :\n\n',count_vec.vocabulary_)
print ('vocabulary:\n\n')
for key,value in count_vec.vocabulary_.items():
    print(key,value)
