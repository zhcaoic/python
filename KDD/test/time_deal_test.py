# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:06:20 2018

@author: zhcao
"""

import pandas as pd

fmax = "D:/python_workspace/kdd/data/nan_deal_data/df_pre_5_dingling_aq.csv"
dataset_max = pd.read_csv(fmax, header = 0,  index_col = 2)
dataset_max = dataset_max.drop([dataset_max.columns[0]], axis = 1)

fin = "D:/python_workspace/kdd/data/nan_deal_data/df_pre_1_aotizhongxin_aq.csv"
dataset = pd.read_csv(fin, header = 0, index_col = 2)
dataset = dataset.drop([dataset.columns[0]], axis = 1)
print(dataset.info())

max_index_list = dataset_max.index.tolist()
index_list = dataset.index.tolist()
print(len(max_index_list))
#print(index_list)

# get mean value
mean_list = dataset.mean().astype('float32').tolist()
PM2_5_mean = mean_list[0]
PM10_mean = mean_list[1]
NO2_mean = mean_list[2]
CO_mean = mean_list[3]
O3_mean = mean_list[4]
SO2_mean = mean_list[5]

# reindex
dataset = dataset.reset_index(drop = True)
print(dataset.info())
dataset = dataset.reindex(max_index_list)
print(dataset.info())
"""
cnt = 0
for index in max_index_list:
    if index in index_list:
        cnt = cnt + 1
    else:
        print(index)
print(cnt)
"""