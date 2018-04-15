# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:14:54 2018

@author: zhcao
"""

import pandas as pd

fname = "E:/KDD/test_data/beijing_17_18_aq.csv"

dataset = pd.read_csv(fname)

print(dataset.info())
