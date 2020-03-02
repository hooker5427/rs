import numpy as np
import xlearn as xl
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import pandas as pd 
import os 


columns ="user_id,movie_id,rating,timestamp,title,genres,gender,age,occupation,zip".split(",")
df = pd.read_csv("movielens_sample.txt"  ,names = columns  ,skiprows=1)

# 去掉 time , title 列
df = df.drop( ["timestamp" ,"title" ],axis  =1 )

df  = df.genres.str.get_dummies(sep ="|").join(df).drop("genres" ,axis =1 )
df  = df.gender.str.get_dummies(sep ="|").join(df).drop("gender" ,axis =1 )
from sklearn.preprocessing import LabelEncoder 
enc = LabelEncoder()
df['zip'] = enc.fit_transform(df['zip'])

#  数据集切分
y  ,X = df.pop('rating') ,df
X_train , X_test ,y_train ,y_test = train_test_split( X ,y ,test_size =0.1)
# DMatrix transition
xdm_train = xl.DMatrix(X_train, y_train)
xdm_test = xl.DMatrix(X_test, y_test)



# Training task
fm_model = xl.create_fm()  # Use factorization machine
# we use the same API for train from file
# that is, you can also pass xl.DMatrix for this API now
fm_model.setTrain(xdm_train)    # Training data
fm_model.setValidate(xdm_test)  # Validation data

# 默认early_stoping ....  
param = {'task':'reg', 'lr':0.2, 
         'lambda':0.002, 
         'metric':'rmse' ,
         'epoch'  :10 , 
        'opt' :'sgd'}
if not os.path.exists("out") : 
    os.mkdir("out")

fm_model.fit(param, './out/model_dm.out')

fm_model.setTest(xdm_test)  # Test data

# Start to predict
# The output result will be stored in output.txt
# if no result out path setted, we return res as numpy.ndarray

y_pred = fm_model.predict("./out/model_dm.out")
print(y_pred)



# 计算rmse 
print('MSE为：',mean_squared_error(y_test,y_pred))
# print('MSE为(直接计算)：',np.mean((y_test-y_pred)**2))
print('RMSE为：',np.sqrt(mean_squared_error(y_test,y_pred)))