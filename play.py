# -*- coding: utf-8 -*-

#%%
import pickle
import time
import numpy as np
from instance_test import ggg, kdata
from DQN import Env, DQN_Agent, ReplayMemory, train, test

#%%
origin = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[0,:]])
destination = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[1,:]])

env = Env(ggg, origin, destination, kdata[2,:], 0)

#%%
def experiment(NUM_EXP, MAX_EPISODE, PUNISHMENT, ALPHA, GAMMA,
               EPS_START, EPS_END, EPS_DECAY, BATCH_SIZE, TARGET_UPDATE, MEMORY_SIZE,
               HIDDEN_DIM1, HIDDEN_DIM2, DEVICE):
    setup_dict = {'num_exp':NUM_EXP, 'max_episodes':MAX_EPISODE, 'punishment':PUNISHMENT, 'alpha':ALPHA, 'gamma':GAMMA, 'eps_start':EPS_START, 'eps_end':EPS_END, 'eps_decay':EPS_DECAY, 'batch_size':BATCH_SIZE, 'target_update':TARGET_UPDATE, 'memory_size':MEMORY_SIZE, 'hidden_dim1':HIDDEN_DIM1, 'hidden_dim2':HIDDEN_DIM2}        
    
    EXP_DATA = []
    file_name1 = time.strftime("%Y%m%d-%H%M%S")
    for j in range(NUM_EXP):
        
        print(j)
        memory = [ReplayMemory(MEMORY_SIZE) for i in range(env.numagent)]
        multi = [DQN_Agent(i, env, memory[i],
                           hidden_dim1 = HIDDEN_DIM1, hidden_dim2 = HIDDEN_DIM2, device = DEVICE, alpha = ALPHA, gamma = GAMMA, batch_size = BATCH_SIZE,
                           eps_start = EPS_START, eps_end = EPS_END, eps_decay = EPS_DECAY)
                    for i in range(env.numagent)]
    
        start = time.time()
        episode_rewards, episode_success, episode_length, best_states, best_actions = train(env, multi, memory, TARGET_UPDATE, MAX_EPISODE, PUNISHMENT, DEVICE)
        end = time.time()
        episode_time = end-start
        
        best_answer = np.max([episode_rewards[i].sum() for i in range(MAX_EPISODE)])
        
        target_policy_answer = sum(test(env, multi, "target"))
#        if target_policy_answer > 0:
#            target_policy_answer -= ARRIVAL_BONUS*env.numagent
        
        parameters = [multi[i].target_net.state_dict() for i in range(env.numagent)]
        
        save = [episode_rewards, episode_success, episode_length, episode_time, best_states, best_actions, best_answer, target_policy_answer, parameters]
        EXP_DATA.append(save)

    file_name2 = time.strftime("%Y%m%d-%H%M%S")
    with open('/home/sle175/rlcombopt/data/%s__%s.p' % (file_name1, file_name2), 'wb') as file:
        pickle.dump(setup_dict, file)
        pickle.dump(EXP_DATA, file)
        
    return
#%% BATCH SIZE
#print("batch size")
#experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
#           ALPHA = 0.00025, GAMMA = 1,
#           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
#           BATCH_SIZE = 64, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
#           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu",
#           GRAPH = ggg, GPH_DATA = kdata)
#%%
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 16, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu")

#%% TARGET_UPDATE
print("target update")
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 20, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu")

experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 5, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu")

#%% MEMORY_SIZE
print("memory size")
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 10, MEMORY_SIZE = 2000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu")

experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 10, MEMORY_SIZE = 500,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu")

#%% NETWORK SIZE
print("network size")
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 30, HIDDEN_DIM2 = 30, DEVICE = "cpu",
           GRAPH = ggg, GPH_DATA = kdata)

experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 90, HIDDEN_DIM2 = 90, DEVICE = "cpu")

#%% ALPHA
print("alpha")
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.000125, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu",
           GRAPH = ggg, GPH_DATA = kdata)

experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.0005, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 32, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu")

#%%
#NUM_EXP = 1
#MAX_EPISODES = 1000
#PUNISHMENT = 5000
#ARRIVAL_BONUS = 0
#
#ALPHA = 0.00025
#GAMMA = 1
#
#EPS_START = 0.1
#EPS_END = 0.1
#EPS_DECAY = 1
#
#BATCH_SIZE = 32
#TARGET_UPDATE = 10
#MEMORY_SIZE = 1000
#
#HIDDEN_DIM1 = 120
#HIDDEN_DIM2 = 120
#
#DEVICE = "cuda"
#
##%%
#origin = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[0,:]])
#destination = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[1,:]])
#
#env = Env(ggg, origin, destination, kdata[2,:], ARRIVAL_BONUS)
#
##%%
#setup_dict = {'num_exp':NUM_EXP, 'max_episodes':MAX_EPISODES, 'punishment':PUNISHMENT, 'arrival_bonus':ARRIVAL_BONUS, 'alpha':ALPHA, 'gamma':GAMMA, 'eps_start':EPS_START, 'eps_end':EPS_END, 'eps_decay':EPS_DECAY, 'batch_size':BATCH_SIZE, 'target_update':TARGET_UPDATE, 'memory_size':MEMORY_SIZE, 'hidden_dim1':HIDDEN_DIM1, 'hidden_dim2':HIDDEN_DIM2}
#
#
##%%
#EXP_DATA = []
#file_name1 = time.strftime("%Y%m%d-%H%M%S")
#for j in range(NUM_EXP):
#    
#    print(j)
#    memory = [ReplayMemory(MEMORY_SIZE) for i in range(env.numagent)]
#    multi = [DQN_Agent(i, env, memory[i],
#                       hidden_dim1 = HIDDEN_DIM1, hidden_dim2 = HIDDEN_DIM2, device = DEVICE, alpha = ALPHA, gamma = GAMMA, batch_size = BATCH_SIZE,
#                       eps_start = EPS_START, eps_end = EPS_END, eps_decay = EPS_DECAY)
#                for i in range(env.numagent)]
#
#    start = time.time()
#    episode_rewards, episode_success, episode_length, best_states, best_actions = train(env, multi, memory, TARGET_UPDATE, MAX_EPISODES, PUNISHMENT, DEVICE)
#    end = time.time()
#    episode_time = end-start
#    
#    best_answer = np.max([episode_rewards[i].sum() for i in range(MAX_EPISODES)])
#    if best_answer > 0:
#        best_answer -= ARRIVAL_BONUS*env.numagent
#    
#    target_policy_answer = sum(test(env, multi, "target"))
#    if target_policy_answer > 0:
#        target_policy_answer -= ARRIVAL_BONUS*env.numagent
#    
#    parameters = [multi[i].target_net.state_dict() for i in range(env.numagent)]
#    
#    save = [episode_rewards, episode_success, episode_length, episode_time, best_states, best_actions, best_answer, target_policy_answer, parameters]
#    EXP_DATA.append(save)
#
##%%
#file_name2 = time.strftime("%Y%m%d-%H%M%S")
#with open('/home/sle175/rlcombopt/data/%s__%s.p' % (file_name1, file_name2), 'wb') as file:
#    pickle.dump(setup_dict, file)
#    pickle.dump(EXP_DATA, file)


#%% NEED FOR PLOTTING
#from scipy.stats import t
#xx = np.arange(1, MAX_EPISODES+1)
#avg = [np.average([EXP_DATA[i][0][j].sum() for i in range(NUM_EXP)]) for j in range(MAX_EPISODES)]
#se = np.array([np.std([EXP_DATA[i][0][j].sum() for i in range(NUM_EXP)], ddof = 1) for j in range(MAX_EPISODES)])/np.sqrt(NUM_EXP)
#tvalue = t.ppf(1-0.025, NUM_EXP-1)
#best = [EXP_DATA[i][5] for i in range(NUM_EXP)]

#%%
#GUROBI = -15459

#%% PLOT 1 
#plt.figure(figsize = (10,6))
#plt.fill_between(xx, avg-tvalue*se,
#                 avg+tvalue*se,
#                 alpha=.3, color = '#69b7d6')
#plt.plot(xx, avg, color = '#69b7d6', label = 'Average Learning Curve of RL')
#plt.axhline(y = GUROBI, color = 'r', label = 'Best Solution found by Gurobi')
#plt.axhline(y = np.max(best), color = 'b', label = 'Best Solution found by RL')
#plt.xlabel('\n Episodes', fontsize = 20)
#plt.ylabel('Total Reward Sum of all agents', fontsize = 20)
#plt.tick_params(axis = 'both', size = 10, labelsize = 12)
#plt.legend(frameon=False, bbox_to_anchor=(0.99, 0.03), loc= "lower right", fontsize = 12)

#%% PLOT 2
#plt.figure(figsize = (10,6))
#plt.axhline(y = GUROBI, color = 'r', label = 'Best Solution found by Gurobi')
#plt.axhline(y = np.average(best), alpha = 0.8, color = '#69b7d6', label = 'Average of Best Solutions found by RL')
#plt.scatter(np.arange(1,NUM_EXP+1), best, color = 'b',
#            label = 'Best Solution found by RL in each experiment')
#plt.xlabel('\n Experiment', fontsize = 20)
#plt.ylabel('Total Reward Sum of all agents', fontsize = 20)
#plt.tick_params(axis = 'both', size = 10, labelsize = 12)
#plt.ylim((min(best)-1200, GUROBI+200))          
#plt.legend(frameon=False, loc= "lower left", fontsize = 12)




