# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:25:21 2018

@author: zhcao
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

# parameter
timesteps = 10

# load predict data & drop stationId
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

# split predict data
pre_values = values_scaled[7000:, :]
pre_x, pre_y = pre_values[:, :-1], pre_values[:, -1]

# reshape input to 3D [samples, timesteps, features]
pre_in = pre_x[0:timesteps, :]
for i in range(1, (pre_x.shape[0] - (timesteps - 1))):
    pre_in = concatenate((pre_in, pre_x[i:i + timesteps, :]), axis = 0)
print(pre_in.shape)
pre_out = pre_y[timesteps - 1:]
print(pre_out.shape)
pre_in = pre_in.reshape(((int(int(pre_in.shape[0]) / timesteps)), timesteps, pre_in.shape[1]))
print(pre_in.shape, pre_out.shape)

# predict
model = load_model("E:/KDD/lstm_data/model/lstm_model_step_10.h5")
pre_result = model.predict(pre_in)

# deal with result
pre_x = pre_x[timesteps - 1:, :]
# invert scaler forecast out
inv_pre = concatenate((pre_x[:, :], pre_result), axis = 1)
inv_pre = scaler.inverse_transform(inv_pre)
value_pre = inv_pre[:, 6]
# invert scaler actual out
pre_out = pre_out.reshape((len(pre_out), 1))
inv_real = concatenate((pre_x[:, :], pre_out), axis = 1)
inv_real = scaler.inverse_transform(inv_real)
value_real = inv_real[:, 6]
print(type(value_real))
print(type(value_pre))
print(value_real.shape, value_pre.shape)

# calculate RMSE
rmse = sqrt(mean_squared_error(value_real, value_pre))
print('PM2.5 RMSE: %.6f' % rmse)

# save result
fo = "E:/KDD/lstm_data/PM2_5_result_1.txt"
time = dataset.index.tolist()
time_test = time[7000 + timesteps:]
fout = open(fo, "w")
fout.write("Time" + "\t" + "PM2.5_REAL" + "\t" + "PM2.5_PRE\n")
for x in range(0, len(time_test)):
    fout.write(str(time_test[x]) + "\t" + str(value_real[x]) + "\t" + str(value_pre[x]) + "\n")
    
fout.close()
