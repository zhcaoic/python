# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:56:28 2018

@author: zhcao
"""

import pandas as pd

fname = "E:/KDD/test_data/df_1_aotizhongxin.csv"
dataset = pd.read_csv(fname, header = 0, index_col = 0)
print(dataset.info())

values = dataset.values
print(type(values))
print(values.shape)

col_name = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
num_list = []
cnt = 0

for j in range(0, 8):
    for i in range(0, values.shape[0]):
        if values[i, j] == values[i, j]:
            cnt = cnt + 1
    print(col_name[j] + ':' + '\t' + str(cnt) + '\n')
    num_list.append(cnt)
    cnt = 0

print(num_list)
