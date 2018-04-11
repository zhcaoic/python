# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:15:32 2018

@author: zhcao
"""

from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA

# array values
dataset = read_csv('D:/python_workspace/LSTM/airplane_data.csv', header = 0, index_col = 0)
values = dataset.values

# pca
pca = PCA(n_components = 30)
new_values = pca.fit_transform(values)
print(new_values.shape)

# gen DataFrame
colnames = list()
for i in range(1, 31):
    colnames += [('VAR%d' % i)]
print(colnames)

new_dataset = DataFrame(new_values)
new_dataset.columns = colnames
new_dataset.to_csv("D:/python_workspace/LSTM/data_pca.csv")

