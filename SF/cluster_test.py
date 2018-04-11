# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:51:59 2018

@author: zhcao
"""

from pandas import read_csv
from sklearn.cluster import KMeans

fo = "D:/python_workspace/SF/test/clu_result_watermelon.txt"

dataset = read_csv('D:/python_workspace/SF/test/watermelon.csv', header = 0, index_col = 0)
values = dataset.values
values = values.astype('float32')

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(values)
result = kmeans.predict(values)

num = dataset.index.tolist()

fout = open(fo, "w")
fout.write("Num" + "\t" + "Clu\n")
for i in range(0, len(num)):
    fout.write(str(num[i]) + "\t" + str(result[i]) + "\n")

fout.close()
