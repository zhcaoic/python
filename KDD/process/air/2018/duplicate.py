# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:18:32 2018

@author: SJTU
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

# open file & get size
initial_size_list = []
size_list = []
for i in range(0, (len(stationId) - 1)):
    filename = "E:/KDD/test_data/2018_data/nan_deal_data/df_pre_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset = pd.read_csv(filename, header = 0, index_col = 2)
    dataset = dataset.drop([dataset.columns[0]], axis = 1)
    # get initial size
    initial_size = dataset.iloc[:, 0].size
    initial_size_list.append(initial_size)
    # get drop duplicate size
    dataset = dataset.drop_duplicates()
    size = dataset.iloc[:, 0].size
    size_list.append(size)
    # print
    print(stationId[i])
    print(str(initial_size) + "\t" + str(size) + "\n")
    # write file
    outf = "E:/KDD/test_data/2018_data/drop_duplicate_data/df_dd_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset.to_csv(outf)
    
min_size = min(size_list)
min_size_index = size_list.index(min(size_list))
print(min_size)
print(min_size_index)

max_size = max(size_list)
max_size_index = size_list.index(max(size_list))
print(max_size)
print(max_size_index)    
