# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 04:37:31 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.

"""
#BLS4Regression
import numpy as np
import scipy.io as scio
from BLS_Regression import bls_regression
NumFea = 6
NumWin = 5
NumEnhan = 41
s = 0.8
C = 2**-30
dataFile = 'F:/MultiAttn-BLS/bls_python/DATASET/AB.mat'
data = scio.loadmat(dataFile)
traindata  = np.double(data['train_x'])
trainlabel = np.double(data['train_y'])
testdata = np.double(data['test_x'])
testlabel = np.double(data['test_y'])

bls_regression(traindata,trainlabel,testdata,testlabel,s,C,NumFea,NumWin,NumEnhan)

