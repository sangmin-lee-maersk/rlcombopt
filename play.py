# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('C:\\Users\\cryin\\Desktop\\Thesis Project\\10_CODE\\DQN_EXP')
from instance_test import ggg, kdata

os.chdir('C:\\Users\\cryin\\Desktop\\Thesis Project\\10_CODE\\DQN_EXP\\global_state\\local_reward')
from DQN import Env, DQN_Agent, ReplayMemory, train, test

MAX_EPISODES = 1000
PUNISHMENT = 0
ARRIVAL_BONUS = 0

BATCH_SIZE = 32
TARGET_UPDATE = 10
MEMORY_SIZE = 1000

HIDDEN_DIM1 = 60
HIDDEN_DIM2 = 60

origin = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[0,:]])
destination = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[1,:]])

env = Env(ggg, origin, destination, kdata[2,:], ARRIVAL_BONUS)

#%%
NUM_EXP = 30

#%%
EXP_DATA = []

for j in range(NUM_EXP):
    
    memory = [ReplayMemory(MEMORY_SIZE) for i in range(env.numagent)]
    multi = [DQN_Agent(i, env, memory[i],
                       hidden_dim1 = HIDDEN_DIM1, hidden_dim2 = HIDDEN_DIM2, batch_size = BATCH_SIZE,
                       eps_start = 0.9, eps_end = 0.05, eps_decay = 200)
                for i in range(env.numagent)]

    episode_rewards, episode_success, episode_length, best_states, best_actions = train(env, multi, memory, TARGET_UPDATE, MAX_EPISODES, PUNISHMENT)
    
    best_answer = np.max([episode_rewards[i].sum() for i in range(MAX_EPISODES)])
    if best_answer > 0:
        best_answer -= ARRIVAL_BONUS*env.numagent
    
    target_policy_answer = sum(test(env, multi, "target"))
    if target_policy_answer > 0:
        target_policy_answer -= ARRIVAL_BONUS*env.numagent
    
    parameters = [multi[i].target_net.state_dict() for i in range(env.numagent)]
    
    save = [episode_rewards, episode_success, episode_length, best_states, best_actions, best_answer, target_policy_answer, parameters]
    EXP_DATA.append(save)

#%% NEED FOR PLOTTING
from scipy.stats import t
xx = np.arange(1, MAX_EPISODES+1)
avg = [np.average([EXP_DATA[i][0][j].sum() for i in range(NUM_EXP)]) for j in range(MAX_EPISODES)]
se = np.array([np.std([EXP_DATA[i][0][j].sum() for i in range(NUM_EXP)], ddof = 1) for j in range(MAX_EPISODES)])/np.sqrt(NUM_EXP)
tvalue = t.ppf(1-0.025, NUM_EXP-1)
best = [EXP_DATA[i][5] for i in range(NUM_EXP)]

#%%
GUROBI = -15459

#%% PLOT 1 
plt.figure(figsize = (10,6))
plt.fill_between(xx, avg-tvalue*se,
                 avg+tvalue*se,
                 alpha=.3, color = '#69b7d6')
plt.plot(xx, avg, color = '#69b7d6', label = 'Average Learning Curve of RL')
plt.axhline(y = GUROBI, color = 'r', label = 'Best Solution found by Gurobi')
plt.axhline(y = np.max(best), color = 'b', label = 'Best Solution found by RL')
plt.xlabel('\n Episodes', fontsize = 20)
plt.ylabel('Total Reward Sum of all agents', fontsize = 20)
plt.tick_params(axis = 'both', size = 10, labelsize = 12)
plt.legend(frameon=False, bbox_to_anchor=(0.99, 0.03), loc= "lower right", fontsize = 12)

#%% PLOT 2
plt.figure(figsize = (10,6))
plt.axhline(y = GUROBI, color = 'r', label = 'Best Solution found by Gurobi')
plt.axhline(y = np.average(best), alpha = 0.8, color = '#69b7d6', label = 'Average of Best Solutions found by RL')
plt.scatter(np.arange(1,NUM_EXP+1), best, color = 'b',
            label = 'Best Solution found by RL in each experiment')
plt.xlabel('\n Experiment', fontsize = 20)
plt.ylabel('Total Reward Sum of all agents', fontsize = 20)
plt.tick_params(axis = 'both', size = 10, labelsize = 12)
plt.ylim((min(best)-1200, GUROBI+200))          
plt.legend(frameon=False, loc= "lower left", fontsize = 12)




