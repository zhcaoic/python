# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:16:41 2018

@author: zhcao
"""

inf_1 = "D:/python_workspace/LSTM/FTPD-C919-10101-PD-171025-G-01-01VIB-ANA001-8192.txt"
inf_2 = "D:/python_workspace/LSTM/FTPD-C919-10101-PD-171025-G-01-02VIB-ANA001-8192.txt"
inf_3 = "D:/python_workspace/LSTM/FTPD-C919-10101-PD-171025-G-01-03VIB-ANA001-8192.txt"
inf_4 = "D:/python_workspace/LSTM/FTPD-C919-10101-PD-171025-G-01-04VIB-ANA001-512.txt"

outf_1 = "D:/python_workspace/LSTM/data_1.txt"
outf_2 = "D:/python_workspace/LSTM/data_2.txt"
outf_3 = "D:/python_workspace/LSTM/data_3.txt"
outf_4 = "D:/python_workspace/LSTM/data_4.txt"

def file_Cut(infname, outfname):
    f = open(infname, "r")
    line = f.readline()
    outf = open(outfname, "w")
    for i in range(0, 10001):
        outf.write(line)
        line = f.readline()
        
    f.close()
    outf.close()
        
def data_Connect(fn_1, fn_2, fn_3, fn_4):
    f1 = open(fn_1, "r")
    f2 = open(fn_2, "r")
    f3 = open(fn_3, "r")
    f4 = open(fn_4, "r")
    line1 = f1.readline().strip("\n")
    line2 = f2.readline().strip("\n")
    line3 = f3.readline().strip("\n")
    line4 = f4.readline()
    outfilename = "D:/python_workspace/LSTM/data_sum.txt"
    outfile = open(outfilename, "w")
    for i in range(0, 10001):
        l2 = line2.split("\t", 1)
        l3 = line3.split("\t", 1)
        l4 = line4.split("\t", 1)
        new_line = line1 + "\t" + l2[1] + "\t" + l3[1] + "\t" + l4[1]
        outfile.write(new_line)
        line1 = f1.readline().strip("\n")
        line2 = f2.readline().strip("\n")
        line3 = f3.readline().strip("\n")
        line4 = f4.readline()
        
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    outfile.close()
    
        
file_Cut(inf_1, outf_1)
file_Cut(inf_2, outf_2)
file_Cut(inf_3, outf_3)
file_Cut(inf_4, outf_4)

data_Connect(outf_1, outf_2, outf_3, outf_4)