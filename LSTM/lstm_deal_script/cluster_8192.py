# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:28:57 2018

@author: zhcao
"""

from pandas import read_csv
from sklearn.cluster import KMeans

fo = "D:/python_workspace/SF/clu_data_cut/clu_result_8192_n3.txt"

dataset = read_csv('D:/python_workspace/SF/clu_data_cut/data_cluster_8192.csv', header = 0, index_col = 0)
values = dataset.values
values = values.astype('float32')

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(values)
result = kmeans.predict(values)

time = dataset.index.tolist()

fout = open(fo, "w")
fout.write("Time" + "\t" + "Clu\n")
for i in range(0, len(time)):
    fout.write(str(time[i]) + "\t" + str(result[i]) + "\n")

fout.close()