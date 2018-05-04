# -*- coding: utf-8 -*-
"""
Created on Wed May  2 19:05:48 2018

@author: zhcao
"""

import pandas as pd

fmax = "E:/KDD/test_data/drop_duplicate_data/df_dd_5_dingling_aq.csv"
dataset_max = pd.read_csv(fmax, header = 0,  index_col = 0)

fin = "E:/KDD/test_data/drop_duplicate_data/df_dd_1_aotizhongxin_aq.csv"
dataset = pd.read_csv(fin, header = 0, index_col = 0)

max_index_list = dataset_max.index.tolist()
index_list = dataset.index.tolist()
print(len(max_index_list))
print(len(index_list))

# get mean value
mean_list = dataset.mean().astype('float32').tolist()
PM2_5_mean = round(mean_list[0])
PM10_mean = round(mean_list[1])
NO2_mean = round(mean_list[2])
CO_mean = round(mean_list[3], 1)
O3_mean = round(mean_list[4])
SO2_mean = round(mean_list[5])
print(PM2_5_mean, PM10_mean, NO2_mean, CO_mean, O3_mean, SO2_mean)

# reindex
dataset = dataset.reindex(max_index_list)
print(dataset.info())

# fill mean value
df1 = pd.DataFrame(dataset['stationId'].fillna("aotizhongxin_aq"))
df2 = pd.DataFrame(dataset['PM2.5'].fillna(PM2_5_mean))
df3 = pd.DataFrame(dataset['PM10'].fillna(PM10_mean))
df4 = pd.DataFrame(dataset['NO2'].fillna(NO2_mean))
df5 = pd.DataFrame(dataset['CO'].fillna(CO_mean))
df6 = pd.DataFrame(dataset['O3'].fillna(O3_mean))
df7 = pd.DataFrame(dataset['SO2'].fillna(SO2_mean))

df_temp = pd.concat([df1, df2], axis = 1)
df_temp = pd.concat([df_temp, df3], axis = 1)
df_temp = pd.concat([df_temp, df4], axis = 1)
df_temp = pd.concat([df_temp, df5], axis = 1)
df_temp = pd.concat([df_temp, df6], axis = 1)
new_dataset = pd.concat([df_temp, df7], axis = 1)

new_dataset.to_csv("E:/KDD/test_data/df_reindex_test.csv")



