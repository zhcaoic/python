# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:29:15 2018

@author: zhcao
"""

import pandas as pd
from deal_nan import nan_Deal

fname = "E:/KDD/test_data/beijing_17_18_aq.csv"
sum_data = pd.read_csv(fname, header = 0)

# get name list
stationId = []
station_col = sum_data['stationId']
index_list = station_col.index.tolist()

sta_name = station_col[0]
stationId.append(sta_name)
for i in range(1, len(index_list)):
    temp = station_col[i]
    if temp != sta_name:
        sta_name = temp
        stationId.append(sta_name)

# open file & deal nan
for i in range(0, len(stationId)):
    filename = "E:/KDD/test_data/2018_data/cut_data/df_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset = pd.read_csv(filename, header = 0, index_col = 0)
    df = nan_Deal(dataset)
    df.to_csv("E:/KDD/test_data/2018_data/nan_deal_data/df_pre_" + str(i + 1) + "_" + stationId[i] + ".csv")
