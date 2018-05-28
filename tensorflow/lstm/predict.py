# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:16:44 2018

@author: zhcao
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


timesteps = 10
batch_size = 30
hidden_size = 30
training_steps = 200
learning_rate = 0.1

# read data
fin = "D:/python_workspace/TensorFlow/Demon/data.csv"
dataset = pd.read_csv(fin, header = 0, index_col = 0)

# gen out col-- only predict PM2.5
df = dataset.shift(1)
df_out = pd.DataFrame(dataset['PM2.5'])
df = pd.concat([df, df_out], axis = 1)
df = df.dropna()
df.columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2', 
              'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph',
              'PM2.5_Out']

# preprocess data
values = df.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range = (0, 1))
data_scaled = scaler.fit_transform(values)

# split data to train & valid
train = data_scaled[:6009, :]
valid = data_scaled[6009:7818, :]

# split into input and output
train_x, train_y = train[:, :-1], train[:, -1]
valid_x, valid_y = valid[:, :-1], valid[:, -1]

# reshape input to 3D [samples, timesteps, features]
train_in = train_x[0:timesteps, :]
valid_in = valid_x[0:timesteps, :]
for i in range(1, (train_x.shape[0] - (timesteps - 1))):
    train_in = np.concatenate((train_in, train_x[i:i + timesteps, :]), axis = 0)
for j in range(1, (valid_x.shape[0] - (timesteps - 1))):
    valid_in = np.concatenate((valid_in, valid_x[j:j + timesteps, :]), axis = 0)
print(train_in.shape, valid_in.shape)
train_out = train_y[timesteps - 1:]
valid_out = valid_y[timesteps - 1:]
print(train_out.shape, valid_out.shape)
print(type(train_in), type(train_out))
train_in = train_in.reshape(((int(int(train_in.shape[0]) / timesteps)), timesteps, train_in.shape[1]))
valid_in = valid_in.reshape(((int(int(valid_in.shape[0]) / timesteps)), timesteps, valid_in.shape[1]))
train_out = train_out.reshape((train_out.shape[0], 1))
valid_out = valid_out.reshape((valid_out.shape[0], 1))
print(train_in.shape, train_out.shape, valid_in.shape, valid_out.shape)

# lstm model
def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    outputs, state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
    # outputs.shape = [batch_size, time, hidden_size]
    output = outputs[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn = None)
    
    if not is_training:
        return predictions, None, None
    
    # get loss
    loss = tf.losses.mean_squared_error(labels = y, predictions = predictions)
    
    # get optimizer
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer = "Adam", learning_rate = learning_rate)
    
    return predictions, loss, train_op

def test(sess, test_in, test_out):
    ds = tf.data.Dataset.from_tensor_slices((test_in, test_out))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
    with tf.variable_scope("model2", reuse = True):
        prediction, loss, train_op = lstm_model(X, y, False)
    
    predictions = []
    labels = []
    for n in range(1800):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)
    
    predictions = np.array(predictions).squeeze()
    predictions = predictions.reshape((predictions.shape[0], 1))
    labels = np.array(labels).squeeze()
    labels = labels.reshape((labels.shape[0], 1))
    
    test_x = valid_x[timesteps - 1:, :]
    inv_pre = np.concatenate((test_x[:, :], predictions), axis = 1)
    inv_pre = scaler.inverse_transform(inv_pre)
    value_pre = inv_pre[:, 11]
    inv_real = np.concatenate((test_x[:, :], labels), axis = 1)
    inv_real = scaler.inverse_transform(inv_real)
    value_real = inv_real[:, 11]
    print(type(value_real))
    print(type(value_pre))
    print(value_real.shape, value_pre.shape)
    
    # calculate RMSE
    rmse = sqrt(mean_squared_error(value_real, value_pre))
    print('PM2.5 RMSE: %.6f' % rmse)
    
    # save result
    fo = "D:/python_workspace/TensorFlow/Demon/model2/result.txt"
    time = dataset.index.tolist()
    time_test = time[6018:7818]
    fout = open(fo, "w")
    fout.write("Time" + "\t" + "PM2.5_REAL" + "\t" + "PM2.5_PRE\n")
    for x in range(0, len(time_test)):
        fout.write(str(time_test[x]) + "\t" + str(value_real[x]) + "\t" + str(value_pre[x]) + "\n")  
    fout.close()
    
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "D:/python_workspace/TensorFlow/Demon/model2/model.ckpt")
    test(sess, valid_in, valid_out)


