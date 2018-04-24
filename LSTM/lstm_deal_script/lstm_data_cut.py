# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:42:48 2018

@author: zhcao
"""

fin = "D:/python_workspace/SF/lstm_data/flatFly_data.txt"
fout = "D:/python_workspace/SF/work2/data/lstm_data_10_5.txt"

def file_Cut(infname, outfname):
    f = open(infname, "r")
    line = f.readline()
    outf = open(outfname, "w")
    for i in range(0, 100002):
        outf.write(line)
        line = f.readline()
    
    f.close()
    outf.close()
    
file_Cut(fin, fout)
