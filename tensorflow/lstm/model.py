# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:08:33 2018

@author: zhcao
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np

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

# train 
def train(sess, train_in, train_out):
    # import train data
    ds = tf.data.Dataset.from_tensor_slices((train_in, train_out))
    #ds = ds.repeat(2).batch(batch_size)
    ds = ds.batch(batch_size)
    X, y = ds.make_one_shot_iterator().get_next()
    print(X.shape, y.shape)
    
    # use model
    with tf.variable_scope("model2", reuse = tf.AUTO_REUSE):
        predictions, loss, train_op = lstm_model(X, y, True)
    
    # initial variables
    sess.run(tf.global_variables_initializer())
    for m in range(training_steps):
        state, l = sess.run([train_op, loss])
        print("train steps: " + str(m) + ", loss: " + str(l))
        
    saver = tf.train.Saver()
    saver.save(sess, "D:/python_workspace/TensorFlow/Demon/model2/model.ckpt")
    
with tf.Session() as sess:
    train(sess, train_in, train_out)
    


















