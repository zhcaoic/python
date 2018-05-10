# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:01:14 2018

@author: zhcao
"""

import pandas as pd

fair = "E:/KDD/lstm_data/2017_2018_data/df_1_aotizhongxin_aq.csv"
dataset_air = pd.read_csv(fair, header = 0, index_col = 0)

fwea = "E:/KDD/weather_data/aoti_data/df_weather.csv"
dataset_wea = pd.read_csv(fwea, header = 0, index_col = 0)

dataset = pd.concat([dataset_air, dataset_wea], axis = 1)

dataset = dataset.dropna()
print(dataset.info())

dataset.to_csv("E:/KDD/lstm_data/input_data/df_aotizhongxin_aq.csv")