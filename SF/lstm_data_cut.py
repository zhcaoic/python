# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:42:48 2018

@author: zhcao
"""

fin = "D:/python_workspace/SF/lstm_data/flatFly_data.txt"
fout = "D:/python_workspace/SF/lstm_data/lstm_data_10_6.txt"

def file_Cut(infname, outfname):
    f = open(infname, "r")
    line = f.readline()
    outf = open(outfname, "w")
    for i in range(0, 100001):
        outf.write(line)
        line = f.readline()
    
    f.close()
    outf.close()
    
file_Cut(fin, fout)
