# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:54:33 2018

@author: zhcao
"""

import pandas as pd
from knn import knnFunction

fname = "E:/KDD/test_data/aotizhongxin_nan_deal/df_1_aotizhongxin.csv"
dataset = pd.read_csv(fname, header = 0, index_col = 0)
print(dataset.info())

# drop NO2 nan
df_NO2_temp = dataset.fillna({'NO2':999})
df_NO2 = df_NO2_temp[df_NO2_temp['NO2'] != 999]

# gen index & data df
df_index = df_NO2.iloc[:, [0, 1]]
df_data = df_NO2.iloc[:, 2:8]

# gen parameter
col_name = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
order = [5, 0, 4, 3, 1]

col_x = ['NO2']
index_col_x = [2]

for i in order:
    # gen y
    col_y = col_name[i]
    index_col_y = i
    
    # knn
    df_data = knnFunction(df_data, col_y, col_x, index_col_y, index_col_x)
    
    # gen x
    col_x.append(col_name[i])
    if 'SO2' in col_x:
        col_x.remove('SO2')
        col_x.insert(0, 'SO2')
    if 'O3' in col_x:
        col_x.remove('O3')
        col_x.insert(0, 'O3')
    if 'CO' in col_x:
        col_x.remove('CO')
        col_x.insert(0, 'CO')
    if 'NO2' in col_x:
        col_x.remove('NO2')
        col_x.insert(0, 'NO2')
    if 'PM10' in col_x:
        col_x.remove('PM10')
        col_x.insert(0, 'PM10')
    if 'PM2.5' in col_x:
        col_x.remove('PM2.5')
        col_x.insert(0, 'PM2.5')
        
    
df = pd.concat([df_index, df_data], axis = 1)
print(df.info())
df.to_csv("E:/KDD/test_data/aotizhongxin_nan_deal/df_nan_deal.csv")
        


