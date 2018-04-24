# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:19:08 2018

@author: zhcao
"""

inf_1 = "D:/python_workspace/SF/data/FTPD-C919-10101-PD-170928-F-01-01VIB-ANA001-8192.txt"
inf_2 = "D:/python_workspace/SF/data/FTPD-C919-10101-PD-170928-F-01-02VIB-ANA001-8192.txt"
inf_3 = "D:/python_workspace/SF/data/FTPD-C919-10101-PD-170928-F-01-03VIB-ANA001-8192.txt"

outf_1 = "D:/python_workspace/SF/lstm_data/data_1.txt"
outf_2 = "D:/python_workspace/SF/lstm_data/data_2.txt"
outf_3 = "D:/python_workspace/SF/lstm_data/data_3.txt"

def file_Cut(infname, outfname):
    f = open(infname, "r")
    line = f.readline()
    outf = open(outfname, "w")
    outf.write(line)
    num_line = 0
    while line:
        if num_line in range(38680000, 87833000):
            #outf.write(str(num_line) + "\t")
            outf.write(line)
            line = f.readline()
            num_line = num_line + 1
        else:
            line = f.readline()
            num_line = num_line + 1
    
    f.close()
    outf.close()

def data_Connect(fn_1, fn_2, fn_3):
    f1 = open(fn_1, "r")
    f2 = open(fn_2, "r")
    f3 = open(fn_3, "r")
    line1 = f1.readline().strip("\n")
    line2 = f2.readline().strip("\n")
    line3 = f3.readline()
    outfilename = "D:/python_workspace/SF/lstm_data/flatFly_data.txt"
    outfile = open(outfilename, "w")
    for i in range(0, 49153001):
        l2 = line2.split("\t", 1)
        l3 = line3.split("\t", 1)
        new_line = line1 + "\t" + l2[1] + "\t" + l3[1]
        outfile.write(new_line)
        line1 = f1.readline().strip("\n")
        line2 = f2.readline().strip("\n")
        line3 = f3.readline()
        
    f1.close()
    f2.close()
    f3.close()
    outfile.close()
    
file_Cut(inf_1, outf_1)
file_Cut(inf_2, outf_2)
file_Cut(inf_3, outf_3)

data_Connect(outf_1, outf_2, outf_3)


