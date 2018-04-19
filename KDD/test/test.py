# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:42:08 2018

@author: zhcao
"""

import numpy as np

array = []

array.append([1, 2, 3])
array.append([4, 5, 6])

print(type(array))

data = np.array(array)

print(type(data))
print(data[1,[1,2]])
