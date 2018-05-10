# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:40:37 2018

@author: zhcao
"""

import pandas as pd

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

# concat 2017 & 2018 data
for i in range(0, (len(stationId) - 1)):
    # 2017 data
    fin_1 = "E:/KDD/test_data/drop_duplicate_data/df_dd_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset_1 = pd.read_csv(fin_1, header = 0, index_col = 0)
    dataset_1 = dataset_1.drop([dataset_1.columns[0]], axis = 1)
    
    # 2018 data
    fin_2 = "E:/KDD/test_data/2018_data/drop_duplicate_data/df_dd_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset_2 = pd.read_csv(fin_2, header = 0, index_col = 0)
    dataset_2 = dataset_2.drop([dataset_2.columns[0]], axis = 1)
    
    # concat
    dataset = pd.concat([dataset_1, dataset_2], axis = 0)
    print(dataset.info())
 
    # save to file
    outf = "E:/KDD/lstm_data/2017_2018_data/df_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset.to_csv(outf)
