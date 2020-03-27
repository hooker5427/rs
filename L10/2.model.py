import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from  sklearn.model_selection import  train_test_split 
from  sklearn.preprocessing import OneHotEncoder 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")



# 1.加载数据
train_path ="./train_clean.csv"
train_data = pd.read_csv( train_path )

Y ,X = train_data.pop("target") , train_data 

# 2. 数据集切割
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.5)
X_train_gbdt, X_train_lr, y_train_gbdt, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)


# 3. 构建模型
# 基于GBDT监督变换
n_estimator =50 
grd =  GradientBoostingClassifier(n_estimators=n_estimator,
                                  random_state=10 ,
                                  max_depth=5,
                                  min_samples_leaf=5)
grd.fit(X_train, y_train)

# 得到OneHot编码
grd_enc = OneHotEncoder(categories='auto')
temp = grd.apply(X_train)
np.set_printoptions(threshold=np.inf)  
grd_enc.fit(grd.apply(X_train)[:, :, 0])

grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

# 使用LR进行预测
y_hat = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))


# 4. 计算NE 
NE = (-1) / len(y_hat) * sum(
    ((1 + y_test) / 2 * np.log(y_hat[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_hat[:, 1])))

p = np.sum(y_test == 1) / len(y_test)
entroy = -1 * (p * np.log(p) + (1 - p) * np.log(1 - p))
print (entroy  ,NE )
print("Normalized Cross Entropy on train : " + str(NE / entroy))