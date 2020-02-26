#coding:utf-8
from surprise import Dataset
from surprise import Reader
from surprise import SVD , SVDpp 
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import  os 
from collections import defaultdict
import  numpy as np 


# rating_small 随即进行抽样进行得到的
# data =  pd.read_csv(  "./rating_small.txt" , delimiter=',' , header= None , names= ["user" ,"movie" ,"rating"] ) 


basedir  = "../savedata"
i= 0 
for  rawfile  in   os.listdir(basedir):
    rawfile  = os.path.join(  os.path.abspath(basedir)  ,rawfile   ) 
    if i ==0 :
        data = np.loadtxt(rawfile , dtype = np.int32  , delimiter = ','  ,usecols  = [0,1,2])
    else :
        tempdata = np.loadtxt(rawfile , dtype = np.int32  , delimiter = ','  ,usecols  = [0,1,2])
        data  =  np.concatenate( [data , tempdata] , axis = 0 )
    i+=1

print("开始读入数据....")
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_df(pd.DataFrame(data), reader=reader)
train_set = data.build_full_trainset()
print("数据加载成功....")


#  预测probe数据集
basedir  =r"../probegoodX"
newfilepath_list =[] 
for  file_name in  os.listdir(  basedir) :
    temppath  =   os.path.join(  os.path.abspath(basedir)  ,file_name   ) 
    newfilepath_list.append(temppath )
    
#排序    
sorted( newfilepath_list) 


# NN 控制测试集中文件的格式
NN = 5  
testsets  = pd.DataFrame()
for j in  range(NN):
    probe_csv_path =newfilepath_list[j]
    probedata = pd.read_table(  probe_csv_path ,delimiter= "," ,skiprows=1 , header =None )
    probedata.columns = ["user","item","rating","timestamp"]
    probedata  = probedata.loc[: ,["user","item","rating"] ]
    testsets = pd.concat(   [testsets   , probedata ]  ,axis  =0  )


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n



algo = SVDpp ()
algo.fit( train_set)

predictions = algo.test(testsets.values)
top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])



kf = KFold(n_splits=3)

algo = SVDpp( )
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
	
	
predictions = algo.test(testsets.values)
top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
	
	


from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import GridSearchCV

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6] }

gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())


predictions = algo.test(testsets.values)
top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])





	
	
