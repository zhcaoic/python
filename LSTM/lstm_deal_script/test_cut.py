# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:31:47 2018

@author: zhcao
"""

fin = "D:/python_workspace/SF/rise/rise_data/rise_data.csv"
fout = "D:/python_workspace/SF/predict_test_data/test_data.csv"

def file_Cut(infname, outfname):
    f = open(infname, "r")
    line = f.readline()
    outf = open(outfname, "w")
    outf.write(line)
    
    for i in range(0, 19970000):
        if i in range(0, 19900000):
            line = f.readline()
        else:
            outf.write(line)
            line = f.readline()
    
    """
    for i in range(0, 70000):
        line = f.readline()
        outf.write(line)
    """
    
    f.close()
    outf.close()
    
file_Cut(fin, fout)