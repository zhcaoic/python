# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:26:03 2018

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
        cols.append(df[[12]].shift(-i))
        cols.append(df[[13]].shift(-i))
        cols.append(df[[14]].shift(-i))
        cols.append(df[[15]].shift(-i))
        if i == 0:
            names += [('VAR%d(t)' % (j+1)) for j in range(12, 16)]
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
dataset = read_csv('D:/python_workspace/SF/work2/data/lstm_data_10_6.csv', header = 0, index_col = 0)
values = dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)
# frame to supervised
sup_data = series_To_Supervised(scaled)
print(sup_data.shape)
print(type(sup_data))

# split into train and test sets
new_values = sup_data.values
train = new_values[:30000, :]
test = new_values[30000:, :]
# split into input and output
train_in, train_out = train[:, :-4], train[:, -4:]
test_in, test_out = test[:, :-4], test[:, -4:]
# reshape input to 3D [samples, timesteps, features]
train_in = train_in.reshape((train_in.shape[0], 1, train_in.shape[1]))
test_in = test_in.reshape((test_in.shape[0], 1, test_in.shape[1]))
print(train_in.shape, train_out.shape, test_in.shape, test_out.shape)

# LSTM network
model = Sequential()
model.add(LSTM(50, input_shape = (train_in.shape[1], train_in.shape[2])))
model.add(Dense(4))
model.compile(loss = 'mse', optimizer = 'adam')
# fit network
history = model.fit(train_in, train_out, nb_epoch = 30, batch_size = 100, validation_data = (test_in, test_out), verbose = 0, shuffle = False)
# plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'test')
pyplot.legend()
pyplot.show()

# predict
yhat = model.predict(test_in)
print(type(yhat))
print(yhat.shape)
test_in = test_in.reshape((test_in.shape[0], test_in.shape[2]))
# invert scaler forecast out
inv_yhat = concatenate((test_in[:, :12], yhat), axis = 1)
inv_yhat = concatenate((inv_yhat, test_in[:, 16:]), axis = 1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 12:16]
# invert scaler actual out
test_out = test_out.reshape((len(test_out), 4))
inv_out = concatenate((test_in[:, :12], test_out), axis = 1)
inv_out = concatenate((inv_out, test_in[:, 16:]), axis = 1)
inv_out = scaler.inverse_transform(inv_out)
inv_out = inv_out[:, 12:16]
print(type(inv_out))
print(type(inv_yhat))
print(inv_out.shape, inv_yhat.shape)

# calculate RMSE
rmse_VIB13 = sqrt(mean_squared_error(inv_out[:, 0], inv_yhat[:, 0]))
print('Test RMSE: %.6f' % rmse_VIB13)
rmse_VIB14 = sqrt(mean_squared_error(inv_out[:, 1], inv_yhat[:, 1]))
print('Test RMSE: %.6f' % rmse_VIB14)
rmse_VIB15 = sqrt(mean_squared_error(inv_out[:, 2], inv_yhat[:, 2]))
print('Test RMSE: %.6f' % rmse_VIB15)
rmse_VIB16 = sqrt(mean_squared_error(inv_out[:, 3], inv_yhat[:, 3]))
print('Test RMSE: %.6f' % rmse_VIB16)

# save result
fo = "D:/python_workspace/SF/work2/data/mul_out_result.txt"

time = dataset.index.tolist()
time_test = time[30001:]

fout = open(fo, "w")
fout.write("Time" + "\t" + "VIB13_ACT" + "\t" + "VIB13_PRE" + "\t" + "VIB14_ACT" + "\t" + "VIB14_PRE" + "\t" + "VIB15_ACT" + "\t" + "VIB15_PRE" + "\t" + "VIB16_ACT" + "\t" + "VIB16_PRE\n")
for x in range(0, len(time_test)):
    fout.write(str(time_test[x]) + "\t" + str(inv_out[x, 0]) + "\t" + str(inv_yhat[x, 0]) + "\t" + str(inv_out[x, 1]) + "\t" + str(inv_yhat[x, 1]) + "\t" + str(inv_out[x, 2]) + "\t" + str(inv_yhat[x, 2]) + "\t" + str(inv_out[x, 3]) + "\t" + str(inv_yhat[x, 3]) + "\n")
    
fout.close()
