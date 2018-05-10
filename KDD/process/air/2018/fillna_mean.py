# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:12:29 2018

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
        
# get max index list
fmax = "E:/KDD/test_data/2018_data/drop_duplicate_data/df_dd_22_qianmen_aq.csv"
dataset_max = pd.read_csv(fmax, header = 0,  index_col = 0)
max_index_list = dataset_max.index.tolist()
print(len(max_index_list))

# open file & do reindex & fill nan mean
for i in range(0, (len(stationId) - 1)):  
    # read file
    filename = "E:/KDD/test_data/2018_data/drop_duplicate_data/df_dd_" + str(i + 1) + "_" + stationId[i] + ".csv"
    dataset = pd.read_csv(filename, header = 0, index_col = 0)
    # get fin index list
    index_list = dataset.index.tolist()
    
    # get all mean value
    mean_list = dataset.mean().astype('float32').tolist()
    PM2_5_mean = round(mean_list[0])
    PM10_mean = round(mean_list[1])
    NO2_mean = round(mean_list[2])
    CO_mean = round(mean_list[3], 1)
    O3_mean = round(mean_list[4])
    SO2_mean = round(mean_list[5])
    
    if i == 21:
        df_result = dataset.copy()
        df_result = df_result.reset_index()
        print(df_result.info())
    else:
        # reindex
        dataset = dataset.reindex(max_index_list)
       
        # get nan index
        cnt = 0
        nan_index_list = []
        for index in max_index_list:
            if index in index_list:
                cnt = cnt + 1
            else:
                nan_index_list.append(cnt)
                cnt = cnt + 1
        
        # split df
        new_dataset = dataset.reset_index()
        df_index = new_dataset.iloc[:, 0:2]
        df_values = new_dataset.iloc[:, 2:8]
        
        # fill nan stationId
        df_index = df_index.fillna(stationId[i])
       
        # fill nan value
        values = df_values.values
        values = values.astype('float32')
        for nan_index in nan_index_list:
            if nan_index < 10:
                local_mean_list = values[nan_index - 1, :]
                local_mean_list = local_mean_list.tolist()
                values[nan_index, 0] = local_mean_list[0]
                values[nan_index, 1] = local_mean_list[1]
                values[nan_index, 2] = local_mean_list[2]
                values[nan_index, 3] = local_mean_list[3]
                values[nan_index, 4] = local_mean_list[4]
                values[nan_index, 5] = local_mean_list[5]
            else:
                count = 0
                for j in range(nan_index - 10, nan_index + 10):
                    if j in nan_index_list:
                        count = count + 1
                    else:
                        count = count
                if count == 20:
                    values[nan_index, 0] = PM2_5_mean
                    values[nan_index, 1] = PM10_mean
                    values[nan_index, 2] = NO2_mean
                    values[nan_index, 3] = CO_mean
                    values[nan_index, 4] = O3_mean
                    values[nan_index, 5] = SO2_mean
                else:
                    if ((i == 34) and (nan_index >= 8598)):
                        values[nan_index, 0] = PM2_5_mean
                        values[nan_index, 1] = PM10_mean
                        values[nan_index, 2] = NO2_mean
                        values[nan_index, 3] = CO_mean
                        values[nan_index, 4] = O3_mean
                        values[nan_index, 5] = SO2_mean
                    else:
                        df_split = df_values.iloc[nan_index - 10:nan_index + 10, :]
                        df_temp = df_split.dropna(axis = 0)
                        local_mean_list = df_temp.mean().astype('float32').tolist()
                        
                        values[nan_index, 0] = round(local_mean_list[0])
                        values[nan_index, 1] = round(local_mean_list[1])
                        values[nan_index, 2] = round(local_mean_list[2])
                        values[nan_index, 3] = round(local_mean_list[3], 1)
                        values[nan_index, 4] = round(local_mean_list[4])
                        values[nan_index, 5] = round(local_mean_list[5])
        
        # transmit to dataframe
        columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']    
        df_values = pd.DataFrame(data = values, columns = columns)
       
        # concat
        df_result = pd.concat([df_index, df_values], axis = 1)
        print(df_result.info())
    
    # write data into file
    outf = "E:/KDD/test_data/2018_data/reindex_fillmean_data/df_rf_" + str(i + 1) + "_" + stationId[i] + ".csv"
    df_result.to_csv(outf)
