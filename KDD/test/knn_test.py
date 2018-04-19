# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:52:58 2018

@author: zhcao
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

fname = "E:/KDD/test_data/watermelon_nan.csv"
dataset = pd.read_csv(fname, header = 0, index_col = 0)
#dataset = pd.read_csv(fname)
#print(dataset.info())
        
data = dataset.values
data = data.astype('float32')
   
# gen no nan index list & sum index list
sum_list = dataset.index.tolist()
col = dataset[['VIB1']]
y_col = col.dropna()
y_list = y_col.index.tolist()

# gen train test X Y array
train_index = []
test_index = []
train_x = []
train_y = []
test_x = []

for i in sum_list:
    if i in y_list:
        train_index.append(i)
        train_x.append(data[i-1, 1])
        train_y.append(data[i-1, 0])
    else:
        test_index.append(i)
        test_x.append(data[i-1, 1])
        
# gen df
fill_x = pd.DataFrame(test_x, index = test_index, columns = ['VIB2'])
print(type(fill_x))
print(fill_x) 
        
# reshape train data
train_x = np.array(train_x)
train_x = train_x.reshape((train_x.shape[0], 1))
test_x = np.array(test_x)
test_x = test_x.reshape((test_x.shape[0], 1))

# knn regressor
knn = KNeighborsRegressor(n_neighbors = 3)    
knn.fit(train_x, train_y)
test_y = knn.predict(test_x)
test_y = test_y.astype('float32')

fill_y = pd.DataFrame(test_y, index = test_index, columns = ['VIB1'])

print(type(fill_y))
print(fill_y)  
   
# connect data
fill_df = pd.concat([fill_y, fill_x], axis = 1)
print(type(fill_df))
print(fill_df)

drop_df = dataset.dropna()

df = pd.concat([drop_df, fill_df], axis = 0)
df = df.sort_index()
print(df)

            
            
            
    
    
    
