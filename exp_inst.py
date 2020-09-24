# -*- coding: utf-8 -*-
#%%
import numpy as np
import igraph as ig
import pickle
import time
from DQN import Env, DQN_Agent, ReplayMemory, train, test

#%%
def instance(file):
    gdata = np.genfromtxt(file, dtype=int, skip_header=2, invalid_raise = False).T
    kdata = np.genfromtxt(file, dtype=int, skip_header=len(gdata.T)+2, usecols = (0,1,2)).T
    np.savetxt('tmp.txt', gdata.T[:,0:2], fmt = '%i')
    ggg = ig.Graph.Read_Ncol('tmp.txt', directed = True)
    ggg.vs["name"] = [int(i) for i in ggg.vs["name"]]
    ggg.es["unit cost"] = gdata[2]
    ggg.es["fixed cost"] = gdata[4]
    ggg.es["capacity"] = gdata[3]
    
    return (ggg, kdata)

#%%
def experiment(NUM_EXP, MAX_EPISODE, PUNISHMENT, ALPHA, GAMMA,
               EPS_START, EPS_END, EPS_DECAY, BATCH_SIZE, TARGET_UPDATE, MEMORY_SIZE,
               HIDDEN_DIM1, HIDDEN_DIM2, DEVICE, file):
    ggg = instance(file)[0]
    kdata = instance(file)[1]
    
    origin = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[0,:]])
    destination = np.array([ggg.vs.select(name = i).indices[0] for i in kdata[1,:]])

    env = Env(ggg, origin, destination, kdata[2,:], 0)
    
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

#%%
print('c25_100_10_F_T_5')
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 64, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu", file = 'c25_100_10_F_T_5.dow')

#%%
print('c25_100_10_V_L_5')
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 64, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu", file = 'c25_100_10_V_L_5.dow')
#%%
print('c25_100_30_F_L_5')
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 64, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu", file = 'c25_100_30_F_L_5.dow')
#%%
print('c25_100_30_F_T_5')
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 64, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu", file = 'c25_100_30_F_T_5.dow')
#%%
print('c25_100_30_V_T_5')
experiment(NUM_EXP = 10, MAX_EPISODE = 1000, PUNISHMENT = 5000,
           ALPHA = 0.00025, GAMMA = 1,
           EPS_START = 0.1, EPS_END = 0.1, EPS_DECAY = 1,
           BATCH_SIZE = 64, TARGET_UPDATE = 10, MEMORY_SIZE = 1000,
           HIDDEN_DIM1 = 60, HIDDEN_DIM2 = 60, DEVICE = "cpu", file = 'c25_100_30_V_T_5.dow')
