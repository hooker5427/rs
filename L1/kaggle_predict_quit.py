import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.preprocessing import StandardScaler  # Z_score 分布进行标准化
from sklearn.model_selection import GridSearchCV  # 网格搜索,自动调优
from sklearn.svm import SVC  # SVM 分类
from sklearn.decomposition import PCA  # 降维处理
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.model_selection import KFold
from sklearn import metrics  # 模型评估
from sklearn.ensemble import AdaBoostClassifier

# 载入训练数据
path1 = "train.csv"
train_data = pd.read_csv(path1)

# Y特征数字化
Y = pd.Categorical(train_data["Attrition"]).codes


# precessing

# age  bins
def deal_age(row):
    age = row['Age']
    return int(age / 10)


X = train_data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Attrition'], axis=1)
X = X.drop(['user_id'], axis=1)
label_encode_list = ["BusinessTravel", 'Department', 'EducationField', 'Gender',
                     'JobRole', "MaritalStatus", 'Over18', 'OverTime']
newdf1 = pd.get_dummies(X[label_encode_list], prefix="label_")
X = pd.concat([X, newdf1], axis=1)
X = X.drop(label_encode_list, axis=1)
X["Age"] = X.apply(deal_age, axis=1)

features = X.columns
scaler = StandardScaler()  # 标准化
scaler = scaler.fit_transform(X[features])
scaler = pd.DataFrame(scaler)
X = pd.concat([X, scaler], axis=1)
X = X.drop(X[features], axis=1)
X = X.values

# 预测数据
path2 = r"test.csv"
test_data = pd.read_csv(path2)
test_data = test_data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1)
user_id = test_data['user_id']
x_test = test_data.drop(['user_id'], axis=1)
label_encode_list = ["BusinessTravel", 'Department', 'EducationField', 'Gender',
                     'JobRole', "MaritalStatus", 'Over18', 'OverTime']
newdf1 = pd.get_dummies(x_test[label_encode_list], prefix="label_")
x_test = pd.concat([x_test, newdf1], axis=1)
x_test = x_test.drop(label_encode_list, axis=1)
x_test["Age"] = x_test.apply(deal_age, axis=1)
features = x_test.columns
Scaler = StandardScaler()
Scaler_ = Scaler.fit_transform(x_test[features])
Scaler_ = pd.DataFrame(Scaler_)
x_test = pd.concat([x_test, Scaler_], axis=1)
x_test = x_test.drop(x_test[features], axis=1)

# 数据切割
train_x = X[:700, :]
train_y = Y[:700]
test_x = X[700:, :]
test_y = Y[700:]

# PCA降维处理 
pca = PCA(n_components=0.85, whiten=True)  # 白化
pca.fit(X)
train_x_pca = pca.transform(train_x)
test_x_pca = pca.transform(test_x)
# pca.fit(x_test)
x_test_pca = pca.transform(x_test)


# 模型训练参数 自动调优
def run_cv(train_x, train_y, test_x, test_y, model, **kwargs):
    seed = 7
    cv = KFold(n_splits=5, shuffle=False,
               random_state=seed)
    param_grid = kwargs
    grid_search = GridSearchCV(model(), param_grid, cv=cv, verbose=1)  # 训练自动循环早最优化的参数
    grid_search.fit(train_x, train_y)  # 训练数据
    best_parameters = grid_search.best_estimator_.get_params()  # 寻找最优的参数
    res_dict = {}
    for k, _ in kwargs.items():
        res_dict[k] = best_parameters[k]

    mx1 = model(**res_dict )  # 选择最优参数
    mx1.fit(train_x, train_y)  # 训练
    y_pre = mx1.predict(test_x )
    # fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pre)
    auc =metrics.roc_auc_score(test_y ,y_pre )
    print( "%s score is %s " % (str(model), auc), "best param", res_dict)


run_cv(train_x_pca, train_y, test_x_pca, test_y,
       LogisticRegression, C=np.linspace(0.01, 10, 10) )

run_cv(train_x_pca, train_y, test_x_pca, test_y, SVC,
            C = np.linspace(0.0001, 100, 20),gamma = [0.01, 0.001, 0.0001])

run_cv(train_x_pca, train_y, test_x_pca, test_y, DecisionTreeClassifier,
                max_depth = [3, 4, 5, 6, 7, 8],
                min_samples_leaf = [1, 2, 3, 4]
       )

run_cv(train_x_pca, train_y, test_x_pca, test_y, KNeighborsClassifier, n_neighbors=[3, 4, 5, 6, 7, 8] )

# 测得出最好的模型和参数
mx = LogisticRegression(C=1.12)
mx.fit(train_x_pca, train_y   )
y_hat = mx.predict(x_test_pca )
y_hat = mx.predict_proba(x_test_pca)
y_hat = y_hat[:, 1]
df = pd.DataFrame()
df["user_id"] = user_id
df['Attrition'] = y_hat
df.to_csv("res3.csv", index=True)

