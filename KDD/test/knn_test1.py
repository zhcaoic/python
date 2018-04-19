# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:26:08 2018

@author: zhcao
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

fname = "E:/KDD/test_data/df_NO2_test.csv"
dataset = pd.read_csv(fname, header = 0, index_col = 0)
print(dataset.info())

# gen index & data df
df_index = dataset.iloc[:, [0, 1]]
df_data = dataset.iloc[:, 2:8]

# cut data
df_data_undo = df_data.iloc[:, [0, 1, 3, 4]]
df_data_do = df_data.iloc[:, [2, 5]]

# knn
data = df_data_do.values
data = data.astype('float32')
print(data.shape)

sum_list = dataset.index.tolist()
col = df_data_do[['SO2']]
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
        train_x.append(data[cnt, 0])
        train_y.append(data[cnt, 1])
        cnt = cnt + 1
    else:
        test_index.append(i)
        test_x.append(data[cnt, 0])
        cnt = cnt + 1

# gen df
fill_x = pd.DataFrame(test_x, index = test_index, columns = ['NO2']) 
        
# reshape train data
train_x = np.array(train_x)
train_x = train_x.reshape((train_x.shape[0], 1))
test_x = np.array(test_x)
test_x = test_x.reshape((test_x.shape[0], 1))

# knn regressor
knn = KNeighborsRegressor(n_neighbors = 5)    
knn.fit(train_x, train_y)
test_y = knn.predict(test_x)
test_y = test_y.astype('int')

# connect data
fill_y = pd.DataFrame(test_y, index = test_index, columns = ['SO2'])
fill_df = pd.concat([fill_x, fill_y], axis = 1)

drop_df = df_data_do.dropna()

df = pd.concat([drop_df, fill_df], axis = 0)
df = df.sort_index()
#df.to_csv("E:/KDD/test_data/df_NO2_SO2_test.csv")

df_sum = pd.concat([df_data_undo, df], axis = 1)
print(df_sum.info())



