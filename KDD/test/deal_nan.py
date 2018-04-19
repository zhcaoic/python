# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:56:28 2018

@author: zhcao
"""

import pandas as pd
from knn import knnFunction

def nan_Deal(dataset):
    # gen num list
    values = dataset.values
    print(values.shape)
    
    num_list = []
    count = 0
    for y in range(0, 8):
        for x in range(0, values.shape[0]):
            if values[x, y] == values[x, y]:
                count = count + 1
        num_list.append(count)
        count = 0
    print(num_list)

    # check error
    if num_list[0] != values.shape[0]:
        print('Warning!!! stationId is ' + str(num_list[0]) + '\t' + 'not equal to the len of dataframe' + str(values.shape[0]) + '\n')
    if num_list[1] != values.shape[0]:
        print('Warning!!! utc_time is ' + str(num_list[1]) + '\t' + 'not equal to the len of dataframe' + str(values.shape[0]) + '\n')
    
    # get max col name
    col_name = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    data_num_list = num_list[2:8]

    index_max = data_num_list.index(max(data_num_list))
    col_name_max = col_name[index_max]
    
    # drop max col NaN & gen new dataframe
    dataset_dropna = dataset[dataset[col_name_max] == dataset[col_name_max]]
    
    # gen index & data dataframe
    df_index = dataset_dropna.iloc[:, [0, 1]]
    df_data = dataset_dropna.iloc[:, 2:8]
    
    # gen parameter
    col_x = [col_name_max]
    index_col_x = [index_max]
    
    temp_list = num_list[2:8]
    temp_list.sort(reverse = True)
    order_list = []
    for num in temp_list:
        order_list.append(data_num_list.index(num))
    order = order_list[1:6]
    
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
    
    return df
    
    
    
    
    
    
    
    
    