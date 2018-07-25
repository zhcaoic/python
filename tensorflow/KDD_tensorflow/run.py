# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:32:49 2018

@author: zhcao
"""

from config import Config
from data import DataDeal
from model import LSTMModel

def main():
    # create instance of config
    config = Config()
    
    # create instance of datadeal
    datadeal = DataDeal(config)
    #get input data
    train_in, train_out, valid_in, valid_out, scaler, valid_x = datadeal.data_Deal()
    
    # create instance of model
    model = LSTMModel(config)
    # get result
    pre_value, act_value = model.run_Session(train_in, train_out, valid_in, valid_out)
    #model.run_Session(train_in, train_out, valid_in, valid_out)
    
    # calculate RMSE
    value_pre, value_real = datadeal.inv_Scale(scaler = scaler, valid_x = valid_x, data_pre = pre_value, data_act = act_value)
    
    # save result
    with open(config.dir_output + "result_2.txt", "w") as f:
        f.write("PRE" + "\t" + "ACT" + "\n")
        for i in range(len(value_pre)):
            f.write(str(int(value_pre[i])) + "\t" + str(int(value_real[i])) + "\n")


if __name__ == "__main__":
    main()