# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:34:15 2018

@author: zhcao
"""

import pandas as pd
from math import sqrt
from numpy import square

fin = "E:/KDD/weather_data/test.csv"
dataset = pd.read_csv(fin, header = 0, index_col = 0)

lo_aoti = 116.397
la_aoti = 39.982

values = dataset.values
distance_list = []
point_list = []
for i in range(0, values.shape[0]):
    if i < 5:
        lo_in = values[i, 1]
        la_in = values[i, 2]
        distance = sqrt(square(abs(lo_in - lo_aoti)) + square(abs(la_in - la_aoti)))
        distance_list.append(distance)
        point_list.append(values[i, 0])
    else:
        lo_in = values[i, 1]
        la_in = values[i, 2]
        distance = sqrt(square(abs(lo_in - lo_aoti)) + square(abs(la_in - la_aoti)))
        max_temp = max(distance_list)
        max_index = distance_list.index(max_temp)
        max_point = point_list[max_index]
        if distance < max_temp:
            distance_list.remove(max_temp)
            point_list.remove(max_point)
            distance_list.append(distance)
            point_list.append(values[i, 0])
              
print(point_list)        
print(distance_list)
        