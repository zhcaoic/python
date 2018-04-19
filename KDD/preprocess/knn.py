# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:25:25 2018

@author: zhcao
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def knnFunction(df_data, col_y, col_x, index_col_y, index_col_x):
    """
    df_data: type:dataframe data
    col_y: type:str col name
    col_x: type:list col name list
    index_col_y: type:int col index
    index_col_x: type:list col index list
    """
    
    n = len(col_x)
    data = df_data.values
    data = data.astype('float32')
    
    sum_list = df_data.index.tolist()
    col = df_data[[col_y]]
    y_col = col.dropna()
    y_list = y_col.index.tolist()
    
    # gen train test X Y array
    train_index = []
    test_index = []
    train_x = []
    train_y = []
    test_x = []
    cnt = 0

    for i in sum_list:
        if i in y_list:
            train_index.append(i)
            train_x.append(data[cnt, index_col_x])
            train_y.append(data[cnt, index_col_y])
            cnt = cnt + 1
        else:
            test_index.append(i)
            test_x.append(data[cnt, index_col_x])
            cnt = cnt + 1
    
    # if len(x) == len(y) & no need to predict nan value      
    if len(test_x) == 0:
        index_col_x.append(index_col_y)
        index_col_x.sort()
        
        df_x = pd.DataFrame(train_x, index = train_index, columns = col_x)
        df_y = pd.DataFrame(train_y, index = train_index, columns = [col_y])
        
        df_fill = pd.concat([df_x, df_y], axis = 1)
        df_fill = df_fill.sort_index()
    
    else:                         
        # gen df
        fill_x = pd.DataFrame(test_x, index = test_index, columns = col_x)
        
        # reshape train data
        train_x = np.array(train_x)
        train_x = train_x.reshape((train_x.shape[0], n))
        test_x = np.array(test_x)
        test_x = test_x.reshape((test_x.shape[0], n))
              
        # knn regressor
        knn = KNeighborsRegressor(n_neighbors = 5)    
        knn.fit(train_x, train_y)
        test_y = knn.predict(test_x)
        
        if col_y == 'CO':
            test_y = test_y.astype('float32')
        else:
            test_y = test_y.astype('int')
            
        # connect data
        fill_y = pd.DataFrame(test_y, index = test_index, columns = [col_y])
        fill_df = pd.concat([fill_x, fill_y], axis = 1)
        
        index_col_x.append(index_col_y)
        index_col_x.sort()
        df_drop = df_data.iloc[:, index_col_x] 
        drop_df = df_drop.dropna()
        
        df_fill = pd.concat([drop_df, fill_df], axis = 0)
        df_fill = df_fill.sort_index()
    
    # continue connect data
    index_col_else = []
    for j in range(0, 6):
        if j in index_col_x:
            index_col_x = index_col_x
        else:
            index_col_else.append(j)
            
    df_else = df_data.iloc[:, index_col_else]
    
    df = pd.concat([df_else, df_fill], axis = 1)
    df = df.sort_index()
            
    # reindex col name
    df_SO2 = df[['SO2']]
    df = df.drop('SO2', axis = 1)
    df.insert(0, 'SO2', df_SO2)
    
    df_O3 = df[['O3']]
    df = df.drop('O3', axis = 1)
    df.insert(0, 'O3', df_O3)
    
    df_CO = df[['CO']]
    df = df.drop('CO', axis = 1)
    df.insert(0, 'CO', df_CO)
    
    df_NO2 = df[['NO2']]
    df = df.drop('NO2', axis = 1)
    df.insert(0, 'NO2', df_NO2)
    
    df_PM10 = df[['PM10']]
    df = df.drop('PM10', axis = 1)
    df.insert(0, 'PM10', df_PM10)
    
    df_PM2_5 = df[['PM2.5']]
    df = df.drop('PM2.5', axis = 1)
    df.insert(0, 'PM2.5', df_PM2_5)
    
    # return df
    return df
    
    
