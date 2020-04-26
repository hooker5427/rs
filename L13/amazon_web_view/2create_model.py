import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import tensorflow.keras as  keras

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

np.random.seed(17)
path = "AMZN.csv"
Data = pd.read_csv(path, header=0, usecols=['Date', 'Close'], parse_dates=True, index_col='Date')
print(Data.info())
print(Data.head())
plt.plot(Data)
plt.title("股价日期走势图")
plt.show()  #
'''
Datapch = Data.pct_change()
LogReturn = np.log(1 + Datapch)
# plt.plot(LogReturn)
# plt.show() #逻辑回归走势图
'''

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # 数据放大缩小
DataScaled = scaler.fit_transform(Data)
TrainLen = int(len(DataScaled) * 0.7)
TestLen = len(DataScaled) - TrainLen
TrainData = DataScaled[:TrainLen, :]
TestData = DataScaled[TrainLen:, :]


def DatasetCreation(dataset, TimeStep=1):
    DataX, DataY = [], []
    for i in range(len(dataset) - TimeStep - 1):
        a = dataset[i:(i + TimeStep), 0]
        DataX.append(a)
        DataY.append(dataset[i + TimeStep, 0])
    return np.array(DataX), np.array(DataY)


TimeStep = 3
TrainX, TrainY = DatasetCreation(TrainData, TimeStep)
TestX, TestY = DatasetCreation(TestData, TimeStep)

TrainX = np.reshape(TrainX, (TrainX.shape[0], 1, TrainX.shape[1]))
TestX = np.reshape(TestX, (TestX.shape[0], 1, TestX.shape[1]))


def build_model():
    model = Sequential()
    model.add(LSTM(256, input_shape=(1, TimeStep)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    model.fit(TrainX, TrainY, epochs=1, batch_size=1)
    model.summary()
    return model


import os

model_path = "amazon-new.h5"
if os.path.exists(model_path):
    model = keras.models.load_model( model_path)
else :
    model =build_model()

#


def predict(mode, train ,test ):
    train_test_pred = model.predict(np.concatenate([train, test], axis=0))
    train_test_pred = train_test_pred.raval()
    return train_test_pred




def plot_image(source_data, train, test):
    plt.figure()
    plt.plot(source_data, color="blue", label="true")

    train_test_pred = model.predict(np.concatenate([train, test], axis=0))
    train_test_pred = train_test_pred.ravel()

#     print ( train_test_pred.ravel())

    plt.plot(train_test_pred, color="red", label="predict")
    plt.axvline(len(train), ls="-", color="black")
    plt.title("amazon stock predict ")
    plt.legend(loc="upper left ")
    plt.savefig("stock.png")
    plt.show()



plot_image(DataScaled, TrainX, TestX)
