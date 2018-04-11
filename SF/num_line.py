# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:19:06 2018

@author: zhcao
"""

def file_Search(fname):
    f = open(fname,"r")
    line = f.readline()
    num_line = 0
    while line:
        line = f.readline()
        num_line = num_line + 1
    print(num_line)
    
    f.close()
    
file_Search("D:/python_workspace/SF/data/FTPD-C919-10101-PD-170928-F-01-03VIB-ANA001-8192.txt")