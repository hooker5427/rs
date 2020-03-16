from surprise import KNNWithMeans
from surprise import Dataset, Reader
import pandas as pd 
from surprise.model_selection import KFold
from surprise import accuracy
from  collections import defaultdict 
import pprint
# 数据读取
path ='./movielens_sample.txt'
df =  pd.read_csv(path , usecols = [0 ,1 ,2] ,skiprows= 1 )
df.columns = [ 'user' ,'item'  , 'rating']
reader = Reader(line_format='user item rating', sep=',' )
data = Dataset.load_from_df(df, reader=reader)
trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个

kf = KFold(n_splits=5)
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    print (rmse,  rmse * rmse )


predictions= []
for row in df.itertuples() :
    user ,  item = getattr(row, 'user'), getattr(row, 'item') 
    predictions.append ( [  user ,  item  , algo.predict(  user, item ).est]) 
    
print ("*"*100)
print ("user\titem\tpredict\n")
pprint.pprint(predictions )
