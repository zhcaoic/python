# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 16:33:21 2018

@author: zhcao
"""

import os
import logging

def get_Logger(filename):
    """Return a logger instance that writes in filename
    
    Args: 
        filename: (string) path to log.txt
    
    Returns:
        logger: instance of logger
        
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format = '%(message)s', level = logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s:%(message)s'))
    logging.getLogger().addHandler(handler)
    
    return logger


class Config():
    def __init__(self):
        """Initialize hyperparameters
        
        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        
        # create instance of logger
        self.logger = get_Logger(self.path_log)
        
    # general config
    dir_output = "D:/python_workspace/TensorFlow/test/output/"
    dir_model = dir_output + "model/model_2.ckpt"
    path_log = dir_output + "log.txt"
    
    # dataset
    path_data = "D:/python_workspace/TensorFlow/test/data.csv"
    
    # training
    nepochs     = 20000
    repeat_num  = 100
    batch_size  = 30
    lr_method   = "Adam"
    lr          = 0.002
    lr_decay    = 0.9
    timesteps   = 10
    hidden_size = 32
    keep_prob   = 1.0
    
    # model hyperparameters
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
