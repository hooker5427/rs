from  sklearn.tree import  DecisionTreeClassifier
import numpy as np


# 不调节参数

# 数据加载
# (train_x, train_y), (test_x, test_y) = mnist.load_data() #从网上下载数据集
data = np.load('../mnist.npz')  # 从本地读取数据集
train_x, train_y, test_x, test_y = data['x_train'], data['y_train'], data['x_test'], data['y_test']  # 数据

# 展示图片
# import matplotlib.pyplot as plt
# (60000, 28, 28)
# plt.imshow( train_x[0])
# plt.show()

train_x = train_x.reshape(train_x.shape[0], 28*28)
test_x = test_x.reshape(test_x.shape[0], 28*28)
train_x = train_x / 255
test_x = test_x / 255

model  = DecisionTreeClassifier()
# 传入训练数据进行训练
model.fit(train_x, train_y )
y_new = model.predict(test_x )
print ( model.score( test_x , test_y ) )
