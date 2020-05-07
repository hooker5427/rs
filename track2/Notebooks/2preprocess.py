import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

train_path ="../train.csv"
train_data = pd.read_csv( train_path  )

test1 = pd.read_csv('../test1.csv')
test1['label'] = 0
train_test = pd.concat( [train_data , test1] ,axis = 0 ).reset_index()
train_test  = train_test[ ['android_id', 'apptype', 'carrier' ,'version' ,
                          'media_id', 'ntt',  'osv','package', 
                           'sid', 'fea_hash', 'location',
                               'fea1_hash', 'cus_type' ,'label']]
train_test.head()

plt.figure(figsize=( 30 ,8))
train_test.osv = train_test.osv.fillna('8.1.0')
train_test.osv.apply(  lambda x: x.replace("Android_" ,"" )).value_counts()\
                                        .plot.bar()


def osv_fun(x ):
    l = [ str(i) for i in range(10)]
    if x[0] in l:
        return x[0]
    else :
        return 'u'
train_test['osv'] = train_test.osv.apply(osv_fun)


def keep_topk( x  , colname  , k = 30 ):
    _cates = train_test[colname].value_counts().index
    u__list = _cates[ -(len(_cates) -k) :]
    
    if  x in u__list:
        return 'u'
    else :
        return x 


from tqdm import tqdm
k = 50  
for  feature in tqdm (['apptype', 'package' ,'version' ,'media_id']):

    _cates = train_test[feature].value_counts().index
    u__list = _cates[ -(len(_cates) -k) :]
    for idx , x in tqdm(enumerate(train_test[feature].values)):
        if  x in u__list:
            train_test[feature][idx] ='u'


    # train_test[feature] = train_test[feature].apply( 
    #                 lambda x :keep_topk( x,feature ,k = 50) )

#     plt.figure(figsize=( 30 ,8))
#     train_data['media_id'].value_counts().plot.bar()
#     sns.countplot(x ='media_id' , hue='label' , data= train_data )                                        


train_test.loc[train_test.carrier ==-1.0 ,'carrier' ] = 0.

train_test.to_csv('train_test.csv' , index =False )

train_test.loc[:499999, :].to_csv("train_lightgbm.csv" ,index =False )
train_test.loc[500000: , :].to_csv("test_lightgbm.csv" ,index =False )