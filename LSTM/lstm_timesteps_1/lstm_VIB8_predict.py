# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:11:43 2018

@author: zhcao
"""

from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

def series_To_Supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('VAR%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence
    # only predict VIB8
    for i in range(0, n_out):
        cols.append(df[[7]].shift(-i))
        if i == 0:
            names += [('VAR%d(t)' % (j+1)) for j in range(7, 8)]
        else:
            names += [('VAR%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # concat
    agg = concat(cols, axis = 1)
    agg.columns = names
    # drop NaN
    if dropnan:
        agg.dropna(inplace = True)
    
    return agg

# load predict data
pre_dataset = read_csv('D:/python_workspace/SF/predict_test_data/09_11_12__20.csv', header = 0, index_col = 0)
pre_values = pre_dataset.values
pre_values = pre_values.astype('float32')
pre_scaler = MinMaxScaler(feature_range = (0, 1))
pre_scaled = pre_scaler.fit_transform(pre_values)
pre_df = series_To_Supervised(pre_scaled)
pre_data = pre_df.values
pre_in, pre_out = pre_data[:, :-1], pre_data[:, -1]
pre_in = pre_in.reshape((pre_in.shape[0], 1, pre_in.shape[1]))
print(pre_in.shape, pre_out.shape)

# predict
model = load_model("D:/python_workspace/SF/work2/data/VIB8_model.h5")
yhat = model.predict(pre_in)
pre_in = pre_in.reshape((pre_in.shape[0], pre_in.shape[2]))
# invert scaler forecast out
inv_yhat = concatenate((pre_in[:, :7], yhat), axis = 1)
inv_yhat = concatenate((inv_yhat, pre_in[:, 8:]), axis = 1)
inv_yhat = pre_scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 7]
# invert scaler actual out
pre_out = pre_out.reshape((len(pre_out), 1))
inv_out = concatenate((pre_in[:, :7], pre_out), axis = 1)
inv_out = concatenate((inv_out, pre_in[:, 8:]), axis = 1)
inv_out = pre_scaler.inverse_transform(inv_out)
inv_out = inv_out[:, 7]
print(type(inv_out))
print(type(inv_yhat))
print(inv_out.shape, inv_yhat.shape)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_out, inv_yhat))
print('VIB8 RMSE: %.6f' % rmse)

# save result
fo = "D:/python_workspace/SF/work2/data/VIB8_tp_1_result.txt"

time = pre_dataset.index.tolist()
time_test = time[1:]

fout = open(fo, "w")
fout.write("Time" + "\t" + "VIB8_ACT" + "\t" + "VIB8_PRE\n")
for x in range(0, len(time_test)):
    fout.write(str(time_test[x]) + "\t" + str(inv_out[x]) + "\t" + str(inv_yhat[x]) + "\n")
    
fout.close()










