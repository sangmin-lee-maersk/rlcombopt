# -*- coding: utf-8 -*-

import numpy as np
import igraph as ig
import os
# os.chdir('C:\\Users\\cryin\\Desktop\\Thesis Project\\10_CODE\\Instances')
#%%
file = 'c25_100_10_F_L_5.dow'
#%%
#file = 'c100_400_30_F_L_10.dow'
#%%
#file = 'c49.dow'
#%%
gdata = np.genfromtxt(file, dtype=int, skip_header=2, invalid_raise = False).T
kdata = np.genfromtxt(file, dtype=int, skip_header=len(gdata.T)+2, usecols = (0,1,2)).T
np.savetxt('tmp.txt', gdata.T[:,0:2], fmt = '%i')
ggg = ig.Graph.Read_Ncol('tmp.txt', directed = True)
ggg.vs["name"] = [int(i) for i in ggg.vs["name"]]
ggg.es["unit cost"] = gdata[2]
ggg.es["fixed cost"] = gdata[4]
ggg.es["capacity"] = gdata[3]
