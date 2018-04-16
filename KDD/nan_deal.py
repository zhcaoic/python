# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:14:54 2018

@author: zhcao
"""

import pandas as pd

fname = "E:/KDD/test_data/aotizhongxin_nan_deal/df_1_aotizhongxin.csv"
dataset = pd.read_csv(fname, header = 0, index_col = 0)
print(dataset.info())

# drop NO2 nan
df_NO2_temp = dataset.fillna({'NO2':999})
df_NO2 = df_NO2_temp[df_NO2_temp['NO2'] != 999]
print(df_NO2.info())
df_NO2.to_csv("E:/KDD/test_data/df_NO2_test.csv")

# fill SO2 NaN

    

