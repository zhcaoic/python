# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:23:01 2018

@author: zhcao
"""

fin = "D:/python_workspace/SF/lstm_data/flatFly_data.txt"
fout = "D:/python_workspace/SF/predict_test_data/test_data.txt"

def file_Cut(infname, outfname):
    f = open(infname, "r")
    line = f.readline()
    outf = open(outfname, "w")
    outf.write(line)
    for i in range(0, 45070000):
        if i in range(0, 45000000):
            line = f.readline()
        else:
            outf.write(line)
            line = f.readline()
    
    f.close()
    outf.close()
    
file_Cut(fin, fout)