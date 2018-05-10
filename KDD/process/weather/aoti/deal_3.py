# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:31:45 2018

@author: zhcao
"""

import pandas as pd

for i in range(1, 6):
    if i == 1:
        fin = "E:/KDD/weather_data/aoti_data/df_" + str(i) + "_weather.csv"
        df = pd.read_csv(fin, header = 0, index_col = 1)
        df = df.drop([df.columns[0]], axis = 1)
    else:
        fin = "E:/KDD/weather_data/aoti_data/df_" + str(i) + "_weather.csv"
        df_temp = pd.read_csv(fin, header = 0, index_col = 1)
        df_temp = df_temp.drop([df_temp.columns[0]], axis = 1)        
        df = pd.concat([df, df_temp], axis = 1)
        
print(df.info())
df.to_csv("E:/KDD/weather_data/aoti_data/df_weather.csv")
