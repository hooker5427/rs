# 使用DCN模型对Avazu CTR进行预估
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deepctr.models import DCN #使用Deep & Cross
from deepctr.inputs import SparseFeat,get_feature_names
import pickle


fp_train_f =r'F:\rs\avazu-ctr-prediction\code\train_features.csv'

##==================== DeepFM 训练 ====================##
data = pd.read_csv(fp_train_f, dtype={'id':str}, index_col=None)
print('data loaded')

#数据加载
sparse_features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'app_domain', 'app_id', 'banner_pos','device_conn_type', 'device_id', 'device_ip', 'device_model', 'device_type', 'hour', 'id', 'site_domain', 'site_id', 'day', 'site_category_0', 'site_category_1', 'site_category_2', 'site_category_3', 'site_category_4', 'site_category_5', 'app_category_0', 'app_category_1']
target = ['click']
# 对特征标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(fixlen_feature_columns)
print(feature_names)

# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

# 使用DCN进行训练
#model = DCN(linear_feature_columns, dnn_feature_columns, task='regression')
model = DCN(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# 使用DCN进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print("test RMSE", rmse)

# 输出LogLoss
from sklearn.metrics import log_loss ,roc_auc_score

print ("auc " , roc_auc_score( test[target].values, pred_ans  ))

score = log_loss(test[target].values, pred_ans)
print("LogLoss", score)
