# -*- coding: utf-8 -*-
"""
Single-Pair Shortest Path Problems
"""

#%%
import os
os.chdir('C:\\Users\\cryin\\Desktop\\Thesis Project\\10_CODE')

#%%
import time
import igraph as ig
import Env_aprx_ as Env_
import Env_aprx as Env
import plot
import numpy as np

#%%
g1 = ig.Graph.Read_Ncol('C:\\Users\\cryin\\Documents\\g.txt', directed = False)
g1 = Env.graph_s(g1, 6, 9, 10, 40)
#%%
Env.answer(g1)

#%%
ex2 = Env.SPSP_Env(g1, 4)
start = time.process_time()
rst2 = ex2.Q_learning(30000, 0.0005, 1)
end = time.process_time()
print(end-start)

#%%
ex2 = Env_.SPSP_Env(g1, 4)
start = time.process_time()
rst2 = ex2.Q_learning(30000, 0.0005, 1)
end = time.process_time()
print(end-start)

#%%
ex2.greedy_list_v(rst2['param'])

#%%
ex2.test(rst2['param'])

#%%
ex2.linear_approx(0,4,rst2['param'])


#%%
plot.plot_reward(rst2['reward'],-1770, 30)
plot.plot_length(rst2['length'],4)


#%%
'''ex3'''
g3 = ig.Graph.Read_Ncol('C:\\Users\\cryin\\Documents\\g3.txt', directed = False)
g3 = Env.graph_s(g3, 99, 96)

#%%
Env.answer(g3)

#%%
ex3 = Env.SPSP_Env(g3, 50)
start = time.process_time()
rst3 = ex3.Q_learning(1000, 0.00005, 1)
end = time.process_time()
print(end-start)

#%%
ex3 = Env_.SPSP_Env(g3, 50)
start = time.process_time()
rst3 = ex3.Q_learning(1000, 0.00005, 1)
end = time.process_time()
print(end-start)

#%%
ex3 = Env.SPSP_Env(g3, 50)

#%%
rst3 = ex3.Q_learning(30000, 0.00005, 1)

#%%
ex3.greedy_list(rst3['param'])

#%%
ex3.test(rst3['param'])

#%%
plot.plot_reward(rst3['reward'], -41000, 1000)
plot.plot_length(rst3['length'], 6)

#%%
'''ex4'''
g4 = ig.Graph.Read_Ncol('C:\\Users\\cryin\\Documents\\g4.txt', directed = False)
g4 = Env.graph_s(g4, 1, 1800)

#%%
Env.answer(g4)

#%%
ex4 = Env.SPSP_Env(g4, 900)

#%%
rst4 = ex4.Q_learning(10000, 0.00005, 1)

#%%
ex3.greedy_list(rst3['param'])

#%%
ex3.test(rst3['param'])

#%%
plot.plot_reward(rst3['reward'], -41000, 1000)
plot.plot_length(rst3['length'], 6)

#%%
tmp = list(ex2.P[1])

#%%

#%%
tmp = ex2.actions(1,np.zeros(25))


