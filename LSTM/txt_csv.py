# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:06:16 2018

@author: zhcao
"""

def formatConvert(filename):
    f = open(filename, "r")
    line = f.readline()
    csvname = "D:/python_workspace/LSTM/data_100.csv"
    fcsv = open(csvname, "w")
    while line:
        items = line.split("\t")
        new_line = items[0]
        for item in items:
            if item == items[0]:
                new_line = new_line
            else:
                new_line = new_line + ',' + item
        fcsv.write(new_line)
        line = f.readline()
    f.close()
    fcsv.close()
    
formatConvert("D:/python_workspace/LSTM/data_sum_100.txt")
    