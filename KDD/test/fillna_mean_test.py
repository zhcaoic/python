# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:42:27 2018

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

# reindex
dataset = dataset.reindex(max_index_list)

# get nan index
cnt = 0
cnt_value = 0
cnt_nan = 0
nan_index_list = []
for index in max_index_list:
    if index in index_list:
        cnt_value = cnt_value + 1
        cnt = cnt + 1
    else:
        nan_index_list.append(cnt)
        cnt_nan = cnt_nan + 1
        cnt = cnt + 1

# split & fill stationId nan
new_dataset = dataset.reset_index()
df_index = new_dataset.iloc[:, 0:2]
df_values = new_dataset.iloc[:, 2:8]
df_index = df_index.fillna("aotizhongxin_aq")

# fill nan value
values = df_values.values
values = values.astype('float32')
for i in nan_index_list:
    df_split = df_values.iloc[i-10:i+10, :]
    df_temp = df_split.dropna(axis = 0)
    mean_list = df_temp.mean().astype('float32').tolist()
    
    values[i, 0] = round(mean_list[0])
    values[i, 1] = round(mean_list[1])
    values[i, 2] = round(mean_list[2])
    values[i, 3] = round(mean_list[3], 1)
    values[i, 4] = round(mean_list[4])
    values[i, 5] = round(mean_list[5])

columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']    
df_values = pd.DataFrame(data = values, columns = columns)
print(df_values.info())

# concat
df_result = pd.concat([df_index, df_values], axis = 1)
print(df_result.info())
df_result.to_csv("E:/KDD/test_data/df_reindex_test_1.csv")



