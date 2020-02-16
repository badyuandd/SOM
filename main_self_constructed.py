# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:26:56 2018

@author: Zhiyuan ZHU
"""

import numpy as np
import pway_funcs as fn2
import scipy.io
import GRAB as grab
from sklearn import preprocessing
import matplotlib.pyplot as plt
from nilearn import plotting

max_iter = 50
tol = 1e-4
dual_max_iter = 600
dual_tol = 1e-5
train = scipy.io.loadmat('data/simulation_data.mat')['Simulation_data_p']
train1 = fn2.standardize(train)
data_1 = train1.T
S = np.cov(data_1)
data = train
node_num = 90
(Theta, blocks) = grab.BCD_modified(Xtrain=data,Ytrain=data,S=S,lambda_1=16,lambda_2=8,K=4,max_iter=max_iter,tol=tol,
                                       dual_max_iter=dual_max_iter,dual_tol=dual_tol)
print ("Theta: ", Theta)
print ("Overlapping Blocks: ", blocks)
Theta = -Theta
Theta = Theta - np.diag(np.diag(Theta))
Theta = Theta.reshape(((node_num*node_num),1))
max_abs_scaler = preprocessing.MaxAbsScaler()
Theta = max_abs_scaler.fit_transform(Theta)
Theta = Theta.reshape((node_num,node_num))      
Theta[Theta!=0] = 1
plotting.plot_matrix(Theta, figure=(9, 7), vmax=1, vmin=-1,)
plt.title("SOM",fontsize='40')       
