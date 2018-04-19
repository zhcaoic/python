# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:27:12 2018

@author: zhcao
"""

import pandas as pd
from knn import knnFunction

fname = "E:/KDD/test_data/df_NO2_test.csv"
dataset = pd.read_csv(fname, header = 0, index_col = 0)
print(dataset.info())

# gen index & data df
df_index = dataset.iloc[:, [0, 1]]
df_data = dataset.iloc[:, 2:8]

col_y = 'SO2'
col_x = ['NO2']
index_col_y = 5
index_col_x = [2]

df_1 = knnFunction(df_data, col_y, col_x, index_col_y, index_col_x)
print(df_1.info())

