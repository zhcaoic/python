# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:19:50 2018

@author: zhcao
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot

# parameter
timesteps = 10

# load data & drop stationId
dataset = pd.read_csv('E:/KDD/lstm_data/input_data/df_rf_1_aotizhongxin_aq.csv', header = 0, index_col = 1)
dataset = dataset.drop([dataset.columns[0]], axis = 1)
dataset = dataset.drop([dataset.columns[0]], axis = 1)

# gen out col-- only predict PM2.5
df = dataset.shift(1)
df_out = pd.DataFrame(dataset['PM2.5'])
df = pd.concat([df, df_out], axis = 1)
df = df.dropna()
df.columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 'PM2.5_Out']

# preprocess data
values = df.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range = (0, 1))
values_scaled = scaler.fit_transform(values)

# split data to train & valid
train = values_scaled[:3000, :]
valid = values_scaled[3000:7000, :]

# split into input and output
train_x, train_y = train[:, :-1], train[:, -1]
valid_x, valid_y = valid[:, :-1], valid[:, -1]

# reshape input to 3D [samples, timesteps, features]
train_in = train_x[0:timesteps, :]
valid_in = valid_x[0:timesteps, :]
for i in range(1, (train_x.shape[0] - (timesteps - 1))):
    train_in = concatenate((train_in, train_x[i:i + timesteps, :]), axis = 0)
for j in range(1, (valid_x.shape[0] - (timesteps - 1))):
    valid_in = concatenate((valid_in, valid_x[j:j + timesteps, :]), axis = 0)
print(train_in.shape, valid_in.shape)
train_out = train_y[timesteps - 1:]
valid_out = valid_y[timesteps - 1:]
print(train_out.shape, valid_out.shape)
train_in = train_in.reshape(((int(int(train_in.shape[0]) / timesteps)), timesteps, train_in.shape[1]))
valid_in = valid_in.reshape(((int(int(valid_in.shape[0]) / timesteps)), timesteps, valid_in.shape[1]))
print(train_in.shape, train_out.shape, valid_in.shape, valid_out.shape)

# LSTM network
model = Sequential()
model.add(LSTM(32, input_shape = (train_in.shape[1], train_in.shape[2])))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')

# fit network
history = model.fit(train_in, train_out, nb_epoch = 200, batch_size = 100, validation_data = (valid_in, valid_out), verbose = 2, shuffle = False)

# plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'valid')
pyplot.legend()
pyplot.show()

# save model
model.save("E:/KDD/lstm_data/model/lstm_model_step_" + str(timesteps) + "_test2" + ".h5")
