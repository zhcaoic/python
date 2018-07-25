# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:12:02 2018

@author: zhcao
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

class DataDeal(object):
    """data deal"""
    
    def __init__(self, config):
        """Defines self.config and self.logger
        
        Args: 
            config: (Config instance) class with hyper parameters
        
        """
        self.config = config
        self.logger = config.logger
    
    def data_Deal(self):
        """Data Deal
        
        Args:
            is_inverse: 1 ==> inverse MinMaxScaler operating
                        0 ==> calculate train_in, train_out, valid_in, valid_out
        
        """
        # read data
        dataset = pd.read_csv(self.config.path_data, header = 0, index_col = 0)
        
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
        valid = data_scaled[6009:, :]
        self.logger.info("The shape of train set is {}.".format(train.shape))
        self.logger.info("The shape of valid set is {}.".format(valid.shape))
        
        # split into input and output
        train_x, train_y = train[:, :-1], train[:, -1]
        valid_x, valid_y = valid[:, :-1], valid[:, -1]

        # reshape input to 3D [samples, timesteps, features]
        train_in = train_x[0:self.config.timesteps, :]
        for i in range(1, (train_x.shape[0] - (self.config.timesteps - 1))):
            train_in = np.concatenate((train_in, train_x[i:i + self.config.timesteps, :]), axis = 0)
        valid_in = valid_x[0:self.config.timesteps, :]
        for j in range(1, (valid_x.shape[0] - (self.config.timesteps - 1))):
            valid_in = np.concatenate((valid_in, valid_x[j:j + self.config.timesteps, :]), axis = 0)
        self.logger.info("Before reshape, the shape of train_in is {}.".format(train_in.shape))
        self.logger.info("Before reshape, the shape of valid_in is {}.".format(valid_in.shape))
        
        train_out = train_y[self.config.timesteps - 1:]
        valid_out = valid_y[self.config.timesteps - 1:]
        self.logger.info("Before reshape, the shape of train_out is {}.".format(train_out.shape))
        self.logger.info("Before reshape, the shape of valid_out is {}.".format(valid_out.shape))
        
        train_in = train_in.reshape(((int(int(train_in.shape[0]) / self.config.timesteps)), self.config.timesteps, train_in.shape[1]))
        train_out = train_out.reshape((train_out.shape[0], 1))
        valid_in = valid_in.reshape(((int(int(valid_in.shape[0]) / self.config.timesteps)), self.config.timesteps, valid_in.shape[1]))
        valid_out = valid_out.reshape((valid_out.shape[0], 1))
        self.logger.info("After reshape, the shape of train_in is {}.".format(train_in.shape))
        self.logger.info("After reshape, the shape of train_out is {}.".format(train_out.shape))
        self.logger.info("After reshape, the shape of valid_in is {}.".format(valid_in.shape))
        self.logger.info("After reshape, the shape of valid_out is {}.".format(valid_out.shape))
        
        return train_in, train_out, valid_in, valid_out, scaler, valid_x
        
    
        
    def inv_Scale(self, scaler, valid_x, data_pre, data_act):
        
        test_x = valid_x[self.config.timesteps - 1:, :]
        inv_pre = np.concatenate((test_x[:, :], data_pre), axis = 1)
        inv_pre = scaler.inverse_transform(inv_pre)
        value_pre = inv_pre[:, 11]
        inv_real = np.concatenate((test_x[:, :], data_act), axis = 1)
        inv_real = scaler.inverse_transform(inv_real)
        value_real = inv_real[:, 11]
        
        self.logger.info("The shape of prediction value is {}.".format(value_pre.shape))
        self.logger.info("The shape of actual value is {}.".format(value_real.shape))
        
        # calculate RMSE
        rmse = sqrt(mean_squared_error(value_real, value_pre))
        self.logger.info('PM2.5 RMSE: %.6f' % rmse)
        
        return value_pre, value_real
        
        
       
        
        
        
        