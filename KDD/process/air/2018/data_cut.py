# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:37:31 2018

@author: zhcao
"""

import pandas as pd

fname = "E:/KDD/KDD_data/beijing_201802_201803_aq.csv"
df = pd.read_csv(fname, header = 0)
print(df.info())

fname1 = "E:/KDD/test_data/beijing_17_18_aq.csv"
dataset = pd.read_csv(fname1, header = 0)
#print(dataset.info())

# get name list
stationId = []
station_col = dataset['stationId']
index_list = station_col.index.tolist()
print(len(index_list))

sta_name = station_col[0]
stationId.append(sta_name)

for i in range(1, len(index_list)):
    temp = station_col[i]
    if temp != sta_name:
        sta_name = temp
        stationId.append(sta_name)
    
print(stationId)

# cut data
cnt = 1
for j in stationId:
    df1 = df[df['stationId'] == j]
    df1.to_csv("E:/KDD/test_data/2018_data/cut_data/df_" + str(cnt) + "_" + j + ".csv")
    cnt = cnt + 1