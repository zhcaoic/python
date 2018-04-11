# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:25:42 2018

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
dataset = read_csv('D:/python_workspace/LSTM/data_pca.csv', header = 0, index_col = 0)
values = dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)
# frame to supervised
sup_data = series_To_Supervised(scaled)

# split into train and test sets
new_values = sup_data.values
train = new_values[:3000, :]
test = new_values[3000:, :]
# split into input and output
train_in, train_out = train[:, :-1], train[:, -1]
test_in, test_out = test[:, :-1], test[:, -1]
# reshape input to 3D [samples, timesteps, features]
train_in = train_in.reshape((train_in.shape[0], 1, train_in.shape[1]))
test_in = test_in.reshape((test_in.shape[0], 1, test_in.shape[1]))
print(train_in.shape, train_out.shape, test_in.shape, test_out.shape)

# LSTM network
model = Sequential()
model.add(LSTM(10, input_shape = (train_in.shape[1], train_in.shape[2])))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer = 'adam')
# fit network
history = model.fit(train_in, train_out, nb_epoch = 50, batch_size = 72, validation_data = (test_in, test_out), verbose = 2, shuffle = False)
# plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'test')
pyplot.legend()
pyplot.show()

# predict
yhat = model.predict(test_in)
test_in = test_in.reshape((test_in.shape[0], test_in.shape[2]))
# invert scaler forecast out
inv_yhat = concatenate((yhat, test_in[:, 1:]), axis = 1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaler actual out
test_out = test_out.reshape((len(test_out), 1))
inv_out = concatenate((test_out, test_in[:, 1:]), axis = 1)
inv_out = scaler.inverse_transform(inv_out)
inv_out = inv_out[:, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_out, inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_out, label = 'actual')
pyplot.plot(inv_yhat, label = 'predict')
pyplot.legend()
pyplot.show()






