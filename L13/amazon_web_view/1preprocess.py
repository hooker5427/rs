import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(17)
path=r"AMZN.csv"
Data=pd.read_csv(path,header=0,usecols=['Date','Close'],parse_dates=True,index_col='Date')
print(Data.info())
print(Data.head())
#plt.plot(Data)
#plt.show() #股价日期走势图
Datapch=Data.pct_change()
LogReturn=np.log(1+Datapch)
#plt.plot(LogReturn)
#plt.show() #逻辑回归走势图

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()#数据放大缩小
DataScaled=scaler.fit_transform(Data)
TrainLen=int(len(DataScaled)*0.7)
TestLen=len(DataScaled)-TrainLen
TrainData=DataScaled[:TrainLen,:]
TestData=DataScaled[TrainLen:,:]

def DatasetCreation (dataset ,TimeStep=1):
    DataX,DataY=[],[]
    for i in range(len(dataset)-TimeStep-1):
        a=dataset[i:(i+TimeStep),0]
        DataX.append(a)
        DataY.append(dataset[i+TimeStep,0])
    return np.array(DataX),np.array(DataY)

TimeStep=1
TrainX,TrainY=DatasetCreation(TrainData,TimeStep)
TestX,TestY=DatasetCreation(TestData,TimeStep)

TrainX=np.reshape(TrainX,(TrainX.shape[0],1,TrainX.shape[1]))
TestX=np.reshape(TestX,(TestX.shape[0],1,TestX.shape[1]))









