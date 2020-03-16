import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from  deepctr.inputs import get_feature_names 
from  deepctr.models import WDL
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import  tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from deepctr.inputs import  SparseFeat ,VarLenSparseFeat , DenseFeat
import warnings
warnings.filterwarnings('ignore')


# 1. 加载数据
path  = './ml-100k.csv'
data = pd.read_csv(path ,encoding = 'utf-8')

sparse_categorical_features = ['user_id', 'movie_id' ,'gender', 'age','occupation', 'zip']

# 2.对特征标签进行编码
for feature in sparse_categorical_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])

# for  feature in sparse_categorical_features :
#     print (data[feature].nunique())

#  3. 处理单值离散特征
feature_columns  = [] 
for  feature in sparse_categorical_features:
    feature_columns.append( SparseFeat(feature ,data[feature].nunique() , embedding_dim = 4 ,use_hash =False  ))


# 4. 处理多值离散特征
#  4.1 生成词表， 
def get_table(data ,feature_name , sep = '|'):
    s =set()
    for  line in  data[feature_name]:
        s.update( str(line).split(sep)) 
    s.add("<pad>")
    return  len(s) ,s 

max_len , table = get_table(data ,'genres' , sep = '|')
# 4.2 生成索引 
index2key ={  k :v   for  k ,v in enumerate( table)}
key2index  = { v:k for  k ,v   in index2key.items()}

# 4.3 padding 
data['genres'] =data['genres'].astype(str).str.split("|").map( lambda line  :[  key2index[x]  for x  in line  ])
genres_list  = pad_sequences(data['genres']  ,maxlen=max_len,dtype=int,padding='post',value= key2index['<pad>'] ) 

varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', max_len ,embedding_dim=4),
                                            maxlen=max_len,
                                            combiner='mean',
                                            weight_name=None)]

feature_columns.extend(varlen_feature_columns)
linear_feature_columns = feature_columns
dnn_feature_columns = feature_columns

feature_names = get_feature_names(linear_feature_columns+dnn_feature_columns)


#  5.将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}
train_model_input['genres'] = genres_list[ :len(train) ,: ]
test_model_input ['genres'] = genres_list[ len(train): ,: ]
target = ['rating']

# callback 
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
callbacks = [EarlyStopping( monitor='val_loss',patience=3, min_delta=1e-2)]


# 6,建立模型
model = WDL(linear_feature_columns, 
            dnn_feature_columns, 
            task='regression' , 
            l2_reg_linear= 1e-5 ,
            l2_reg_embedding= 1e-5 ,
            l2_reg_dnn=0.01 , 
            init_std=0.0001,
            dnn_hidden_units=( 256, 128) ,
            seed=1024,
            dnn_dropout=0,
            dnn_activation='relu',
               )
model.summary()

# 可以进行调优
from tensorflow.keras.optimizers import Adam 
optmizer = Adam(1e-4)
model.compile(optmizer , "mse", metrics=['mse'], )
history = model.fit(train_model_input, 
                    train[target].values,
                    batch_size=256,
                    epochs=20, 
                    verbose=True,
                    validation_split=0.2, )


# 7.绘图展示， 观看模型训练情况
plt.figure( figsize= (8,8 ))
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 15)
    plt.show()
plot_learning_curves(history)


# model.evaluate(test_model_input , testtarget.values , batch_size=256)

# 使用WDL进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print("test RMSE", rmse)