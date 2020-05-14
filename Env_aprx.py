# -*- coding: utf-8 -*-
#%%
import os
os.chdir('C:\\Users\\cryin\\Desktop\\Thesis Project\\10_CODE')
import numpy as np
import random
import Fourier
import itertools

#%%
def graph_s(nw, o, d, min_cost = 10, max_cost = 100):
    vnum = nw.vcount()
    enum = nw.ecount()
    nw.vs["label"] = list(range(vnum))
    nw.vs["attr"] = ["i"]*vnum
    nw.vs[o]["attr"] = 'o'
    nw.vs[d]["attr"] = 'd'
    np.random.seed(1)
    nw.es["unit cost"] = np.random.randint(min_cost, max_cost, enum)
    
    return nw

#%%
def answer(nw):
    o = nw.vs.find(attr = 'o').index
    d= nw.vs.find(attr = 'd').index
    cost = nw.shortest_paths_dijkstra(nw.vs[o], nw.vs[d], weights = "unit cost")[0]
    path = nw.get_shortest_paths(nw.vs[o], nw.vs[d], "unit cost", output="vpath")[0]
    
    return [cost, path]

#%%
'''single-pair shortest path problem'''
class SPSP_Env:
    
    def __init__(self, nw, order, dim = 2):
        self.nw = nw
        self.order = order
        self.dim = dim
        
        num_states = nw.vcount()
        self.P = {state: {action: [(action, self.reward(state, action), False)]
                    for action in self.nw.neighbors(state)}
                        for state in range(num_states)}
        dest_idx = self.nw.vs.find(attr = 'd').index
        
        for state in self.nw.neighbors(dest_idx):
            self.P[state][dest_idx] = [(dest_idx, self.reward(state, dest_idx), True)]
        

        self.num_s = num_states
        
        upper = self.num_s - 1
        self.FB = Fourier.FourierBasis(self.dim, [[0, upper], [0, upper]],
                                       order = self.order)
        
        self.c_norm = np.linalg.norm(self.FB.multipliers, 2, 1)
        self.c_norm[0] = 1
        
    def reward(self, state, action):
        
        return -self.nw.es[self.nw.get_eid(state, action)]["unit cost"]
    
    def feature(self, state, action):
        pair = np.array([state, action])
        basis = self.FB.computeFeatures(pair)
        return basis

    def actions(self, state, param):
        action_list = list(self.P[state])
        
        def fun(a):
            return np.inner(param, self.feature(state, a))
        
        result = map(fun, action_list)
        
        return list(result)

    def Q_learning(self, episodes, basic_alpha, gamma, epsilon = 0.01, prt_ep = 1000):
        
        rewards_list = []
        length_list = []
        
        initial_state = self.nw.vs.find(attr = 'o').index
        
        param_dim = (self.order + 1)**self.dim
        param = np.zeros(param_dim)
        
        alpha = basic_alpha/self.c_norm
        
        for ep in range(episodes):
            
            total_reward = 0
            state = initial_state
            reward  = 0
            done = False
            t = 0
            
            while not done:
                action_list = list(self.P[state])
                if random.uniform(0, 1) < epsilon:
                    action = random.sample(action_list, 1)[0]
                    
                else:
                    q_list = self.actions(state, param)
                    argmax = np.argmax(q_list)
                    action = action_list[argmax]
                    
                next_state, reward, done = self.P[state][action][0]
                
                total_reward += reward
                
                old_q = q_list[argmax]
                target = reward + gamma * np.max(self.actions(next_state, param))
                
                param = param + alpha*(target - old_q)*self.feature(state, action)
                
                if done:
                    rewards_list.append((ep, total_reward))
                    length_list.append(t)
                else:
                     state = next_state
                     t = t + 1
                     
            if ep % prt_ep == 0:
                print(f"Episode: {ep}")
                
        print("Training finished. \n")
        
        result = {"length": length_list,
                  "reward": rewards_list,
                  "param": param,
                  "basic alpha": basic_alpha,
                  "alpha": alpha,
                  "gamma": gamma,
                  "epsilon": epsilon}
        return result

    def greedy_policy(self, state, param):
        
        return list(self.P[state])[np.argmax(self.actions(state, param))]
    
    def greedy_list(self, param):
        
        greedy_list = []
        for i in range(self.num_s):
            greedy_list.append(self.greedy_policy(i, param))
            
        return greedy_list
    
    def test(self, param):
        done = False
        total_cost = 0
        state = self.nw.vs.find(attr = 'o').index
        path = [state]
        rewards = [0]
        greedy_action = self.greedy_list(param)
        
        while not done:
            action = greedy_action[state]
            next_state, reward, done = self.P[state][action][0]
            total_cost -= reward
            rewards.append(reward)
            path.append(next_state)
            state = next_state
        
        result = [path, rewards, total_cost]
        
        return result
    
    def optimal(self, state, action, param):
        feature = self.feature(state, action)
        a = np.inner(param, feature)
        b= self.P[state][action][0][1] + np.max(self.actions(action, param)) 
        
        return [a, b]