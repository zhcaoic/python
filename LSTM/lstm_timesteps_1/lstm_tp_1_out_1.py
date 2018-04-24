# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:33:46 2018

@author: zhcao
"""

from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot
from numpy import concatenate
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
    # only predict VAR1
    for i in range(0, n_out):
        cols.append(df[[0]].shift(-i))
        if i == 0:
            names += [('VAR%d(t)' % (j+1)) for j in range(1)]
        else:
            names += [('VAR%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # concat
    agg = concat(cols, axis = 1)
    agg.columns = names
    # drop NaN
    if dropnan:
        agg.dropna(inplace = True)
    
    return agg

# load data
dataset = read_csv('D:/python_workspace/SF/lstm_data/lstm_data_10_6.csv', header = 0, index_col = 0)
values = dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)
# frame to supervised
sup_data = series_To_Supervised(scaled)

# split into train and test sets
new_values = sup_data.values
train = new_values[:30000, :]
test = new_values[30000:, :]
# split into input and output
train_in, train_out = train[:, :-1], train[:, -1]
test_in, test_out = test[:, :-1], test[:, -1]
# reshape input to 3D [samples, timesteps, features]
train_in = train_in.reshape((train_in.shape[0], 1, train_in.shape[1]))
test_in = test_in.reshape((test_in.shape[0], 1, test_in.shape[1]))
print(train_in.shape, train_out.shape, test_in.shape, test_out.shape)

# LSTM network
model = Sequential()
model.add(LSTM(50, input_shape = (train_in.shape[1], train_in.shape[2])))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
# fit network
history = model.fit(train_in, train_out, nb_epoch = 30, batch_size = 100, validation_data = (test_in, test_out), verbose = 0, shuffle = False)
# plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'test')
pyplot.legend()
pyplot.show()

# load predict data
pre_dataset = read_csv('D:/python_workspace/SF/lstm_data/lstm_test_data_10_6.csv', header = 0, index_col = 0)
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
yhat = model.predict(pre_in)
pre_in = pre_in.reshape((pre_in.shape[0], pre_in.shape[2]))
# invert scaler forecast out
inv_yhat = concatenate((yhat, pre_in[:, 1:]), axis = 1)
inv_yhat = pre_scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaler actual out
pre_out = pre_out.reshape((len(pre_out), 1))
inv_out = concatenate((pre_out, pre_in[:, 1:]), axis = 1)
inv_out = pre_scaler.inverse_transform(inv_out)
inv_out = inv_out[:, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_out, inv_yhat))
print('Test RMSE: %.6f' % rmse)

# save result
fo = "D:/python_workspace/SF/lstm_data/lstm_data_10_6_result.txt"

time = pre_dataset.index.tolist()
time_test = time[1:]

fout = open(fo, "w")
fout.write("Time" + "\t" + "VIB1_ACT" + "\t" + "VIB1_PRE\n")
for x in range(0, len(time_test)):
    fout.write(str(time_test[x]) + "\t" + str(inv_out[x]) + "\t" + str(inv_yhat[x]) + "\n")
    
fout.close()


