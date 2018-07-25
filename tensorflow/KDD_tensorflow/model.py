# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:14:43 2018

@author: zhcao
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

class LSTMModel(object):
    """LSTM Model"""
    
    def __init__(self, config):
        """Defines self.config and self.logger
        
        Args: 
            config: (Config instance) class with hyper parameters
        
        """
        self.config = config
        self.logger = config.logger
        self.sess = None
        
    def lstm_Model(self, X, y, is_training):
        """Defines lstm_Model
        
        Args:
            X: input
            y: output
            is_training: symbol of the operating is train or predict
                         1 ==> train mode
                         0 ==> predict mode
        
        """
        # add logits
        # lstm
        cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
        outputs, state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
        output = outputs[:, -1, :]
        
        # fully connect
        # the number of layers is 7
        """
        layer_1 = tf.contrib.layers.fully_connected(output, 16, activation_fn = None)
        dropout_1 = tf.nn.dropout(layer_1, self.config.keep_prob)
        
        layer_2 = tf.contrib.layers.fully_connected(dropout_1, 16, activation_fn = None)
        dropout_2 = tf.nn.dropout(layer_2, self.config.keep_prob)
        
        layer_3 = tf.contrib.layers.fully_connected(dropout_2, 8, activation_fn = None)
        dropout_3 = tf.nn.dropout(layer_3, self.config.keep_prob)
        
        layer_4 = tf.contrib.layers.fully_connected(dropout_3, 8, activation_fn = None)
        dropout_4 = tf.nn.dropout(layer_4, self.config.keep_prob)
        
        layer_5 = tf.contrib.layers.fully_connected(dropout_4, 4, activation_fn = None)
        dropout_5 = tf.nn.dropout(layer_5, self.config.keep_prob)
        
        layer_6 = tf.contrib.layers.fully_connected(dropout_5, 4, activation_fn = None)
        dropout_6 = tf.nn.dropout(layer_6, self.config.keep_prob)
        
        predictions = tf.contrib.layers.fully_connected(dropout_6, 1, activation_fn = None)
        """
        layer_1 = tf.contrib.layers.fully_connected(output, 8, activation_fn = None)
        dropout_1 = tf.nn.dropout(layer_1, self.config.keep_prob)
        predictions = tf.contrib.layers.fully_connected(dropout_1, 1, activation_fn = None)
        
        # is_training
        if not is_training:
            return predictions, None, None
        
        # add loss
        loss = tf.losses.mean_squared_error(labels = y, predictions = predictions)
        
        # add optimizer
        train_op = tf.contrib.layers.optimize_loss(loss = loss, global_step = tf.train.get_global_step(), learning_rate = self.config.lr, \
                                                   optimizer = self.config.lr_method)
        
        return predictions, loss, train_op
    
    def train(self, sess, train_in, train_out):
        # input data
        ds = tf.data.Dataset.from_tensor_slices((train_in, train_out))
        ds = ds.repeat(self.config.repeat_num).batch(self.config.batch_size)
        X, y = ds.make_one_shot_iterator().get_next()
        
        # use model & get result
        regularizer = tf.contrib.layers.l2_regularizer(0.01)
        with tf.variable_scope("model", regularizer = regularizer, reuse = tf.AUTO_REUSE):
            predictions, loss, train_op = self.lstm_Model(X, y, is_training = True)
        
        # initial variables
        sess.run(tf.global_variables_initializer())
            
        # calculate
        loss_list = []
        for epoch in range(self.config.nepochs):
            #self.logger.info("Epoch {:} out of {:}".format(epoch + 1, \
                    #self.config.nepochs))
            state, l = sess.run([train_op, loss])
            #self.logger.info("The loss is {}".format(l))
            loss_list.append(l)
            
        # plot loss image
        pyplot.plot(loss_list, label = 'loss')
        pyplot.legend()
        pyplot.show()
            
        # save model
        saver = tf.train.Saver()
        saver.save(sess, self.config.dir_model)
        self.logger.info("model save finish!")
             
    def valid(self, sess, valid_in, valid_out):
        # input data
        ds = tf.data.Dataset.from_tensor_slices((valid_in, valid_out))
        ds = ds.batch(1)
        X, y = ds.make_one_shot_iterator().get_next()
        
        # use model & get result
        with tf.variable_scope("model", reuse = True):
            predictions, loss, train_op = self.lstm_Model(X, y, is_training = False)
        
        # calculate
        pre_result = []
        act_result = []
        for i in range(valid_out.shape[0]):
            p, a = sess.run([predictions, y])
            pre_result.append(p)
            act_result.append(a)
        self.logger.info("Get result!")
        
        pre_result = np.array(pre_result).squeeze()
        pre_result = pre_result.reshape((pre_result.shape[0], 1))
        act_result = np.array(act_result).squeeze()
        act_result = act_result.reshape((act_result.shape[0], 1))
        
        return pre_result, act_result
        

    def run_Session(self, train_in, train_out, valid_in, valid_out):
        with tf.Session() as sess:
            self.train(sess, train_in, train_out)
            pre_value, act_value = self.valid(sess, valid_in, valid_out)
            
        return pre_value, act_value
        

       
        
        
        
        
        

        