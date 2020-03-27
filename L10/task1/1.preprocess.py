import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")


'''
不知道为啥concat之后不 work 
'''
# trainpath  ="./sample_train.csv"
# train_data = pd.read_csv( trainpath) 
# all_data = train_data 


testpath ="./train.csv"
test_data = pd.read_csv( testpath) 
all_data = test_data



n = len(all_data)
for feature in list( all_data.columns) :
    if  feature == 'target' :
        continue
    if  feature.endswith("_bin"):
        continue 
    loss_frea =  np.sum(all_data[feature] < 0) / n 
    if loss_frea >0 :
        print (  feature , loss_frea )


all_data.pop('ps_car_05_cat')
all_data.pop('ps_car_03_cat')

def fun( x):
    return  x if x>0 else 0.8 
all_data.ps_reg_03 = all_data.ps_reg_03.apply( fun  )

# 中位数填充
def fun2( x):
    return  x if x>0 else 0.37 
all_data.ps_car_14 = all_data.ps_car_14.apply( fun2  )



mode_pad_list = [
    'ps_ind_02_cat' ,
    'ps_ind_04_cat' ,
    'ps_ind_05_cat' ,
    'ps_car_01_cat' ,
    'ps_car_02_cat',
    'ps_car_07_cat',
    'ps_car_09_cat',
    'ps_car_11'
]
for feature in mode_pad_list :
    v =  all_data[feature].mode()[0]
    print (  feature  , v )
    for  i ,x in  enumerate(all_data[feature].values) :
        if x < 0 :
            all_data[feature][i] = v


# all_data.to_csv("train_clean.csv" , index =None ) 


all_data.to_csv("train_clean_all.csv" , index =None ) 

