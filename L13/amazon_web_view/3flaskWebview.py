import numpy as np
import pandas as pd
from flask import Flask
from jinja2 import Markup

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import tensorflow.keras as  keras



from pyecharts import options as opts
from pyecharts.charts import Line


app = Flask(__name__, static_folder="templates")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  # 数据放大缩小
def create_xy():

    np.random.seed(17)
    path = "AMZN.csv"
    Data = pd.read_csv(path, header=0, usecols=['Date', 'Close'], parse_dates=True, index_col='Date')


    DataScaled = scaler.fit_transform(Data)
    TrainLen = int(len(DataScaled) * 0.7)
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
    return  TrainX, TrainY ,TestX


import os
def build_model(TrainX, TrainY ,TimeStep ):
    model = Sequential()
    model.add(LSTM(256, input_shape=(1, TimeStep)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    model.fit(TrainX, TrainY, epochs=2, batch_size=64 )
    model.summary()
    return model


def predict(model, train ,test ):
    train_test_pred = model.predict(np.concatenate([train, test], axis=0 )  ,verbose= 0 ,batch_size= 64  )
    train_test_pred = train_test_pred.ravel()
    return train_test_pred

def bar_base(title ,predictions) -> Line:
    c = (
        Line()
            .add_xaxis(  range( len(predictions )))
            .add_yaxis( title ,list(predictions) ,is_smooth= True  ,
                        is_symbol_show= True , is_connect_nones= True )

            .set_global_opts(title_opts=opts.TitleOpts(title= title , subtitle="股票预测"))
    )
    return c


@app.route("/<path:code>/")
def index(code):
    model_path = "amazon-new.h5"
    TrainX, TrainY ,TestX = create_xy()
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = build_model(TrainX, TrainY, 3)
    predictions = predict(model,  TrainX, TestX )

    predictions  = scaler.inverse_transform( predictions.reshape(-1 ,1 ) ).ravel().tolist()

    print ( predictions)
    print ( type( predictions ))


    c = bar_base(code , predictions )

    return Markup(c.render_embed( ))


if __name__ == "__main__":
    app.run()
