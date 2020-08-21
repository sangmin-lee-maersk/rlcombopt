# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import numpy_indexed as npi
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

#%%
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity # memory size
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#%%
class DQN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, env):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.env = env
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim1)
        self.fc0 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, self.output_dim)
        
    def forward(self, input_):
        x = input_.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc0(x))
        x = self.fc2(x)
        
        return x
    
#%%
class DQN_Agent():
    
    def __init__(self, agent_idx, env, memory, hidden_dim1, hidden_dim2, alpha = 0.00025, gamma = 0.999,
                 batch_size = 32, eps_start = 0.9, eps_end = 0.05, eps_decay = 200):
        
        self.agent_idx = agent_idx
        self.env = env
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.alpha = alpha
        self.gamma = gamma
        
        self.memory = memory
        self.batch_size = batch_size
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
    
        self.policy_net = DQN(self.env.state_space_dim, self.hidden_dim1, hidden_dim2, self.env.action_space_dim, env)
        self.target_net = DQN(self.env.state_space_dim, self.hidden_dim1, hidden_dim2, self.env.action_space_dim, env)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr = self.alpha)
        self.MSE_loss = nn.MSELoss()
        
    def get_action(self, state, step): # all agents' locations

        if state[0][self.agent_idx] == self.env.num_s:
            
            return self.env.num_e
        
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * step / self.eps_decay)
        
        action_list = self.env.action_available(state, self.agent_idx) #modify
        
        if len(action_list) == 0:
#            print("fail")
#            print("agent #" + str(self.agent_idx) + ": " + str(state[0][self.agent_idx]) + " " + str(state[1][self.env.nw.incident(state[0][self.agent_idx], mode = "OUT")]) + "qtt" + str(self.env.qtt[self.agent_idx]))
                  
            return np.nan
        
        elif(random.uniform(0, 1) < eps_threshold):
            
            return np.random.choice(action_list, 1)[0]
        
        else:
            cst = np.concatenate((state[0], state[1]), axis = None)
            x = torch.from_numpy(cst).float()
            
            with torch.no_grad():
#                alist = self.env.action_available2(x, self.agent_idx) #modify 
                
                return action_list[torch.argmax(self.policy_net.forward(x)[0][action_list]).item()]
            
    def get_greedy_action(self, state, c): # all agents' locations
        
        if state[0][self.agent_idx] == self.env.num_s:
            
            return self.env.num_e
        
        action_list = self.env.action_available(state, self.agent_idx) #modify
        
        if len(action_list) == 0:
#            print("fail")
#            print("idx " + str(self.agent_idx) + ": " + str(state[0][self.agent_idx]) + " " + str(state[1][self.env.nw.incident(state[0][self.agent_idx], mode = "OUT")]) + "qtt" + str(self.env.qtt[self.agent_idx]))

            return np.nan

        else:
            cst = np.concatenate((state[0], state[1]), axis = None)
            x = torch.from_numpy(cst).float()

            with torch.no_grad():

                if c == "target":
                    tmp = self.target_net.forward(x)
                else:
                    tmp = self.policy_net.forward(x)
               
                return action_list[torch.argmax(tmp[0][action_list]).item()]
            
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state).view(self.batch_size, -1)
        
        non_final_mask = [not c for c in batch.terminal]
        non_final_next_states = [batch.next_state[i] for i in np.where(non_final_mask)[0]]
        
        state_batch = torch.cat([self.policy_net.forward(state_batch[i]) for i in range(self.batch_size)])
        action_batch = torch.from_numpy(np.asarray(batch.action)[:,self.agent_idx])
        reward_batch = torch.from_numpy(np.asarray(batch.reward)[:,self.agent_idx])
        
        state_action_values = state_batch.gather(1,action_batch.long().view(self.batch_size,-1))
        leng = len(non_final_next_states)
        
        if leng != 0:
            max_Q = torch.cat([torch.max(self.target_net.forward(non_final_next_states[i])[0][self.env.action_available2(non_final_next_states[i], self.agent_idx)]).unsqueeze(0) for i in range(leng)])
        
        next_state_values = torch.zeros(self.batch_size)
        
        if leng != 0:
            next_state_values[non_final_mask] = max_Q
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.long()
        
        loss = self.MSE_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        
#%%
class Env:
    
    def __init__(self, network, origins, destinations, quantity, arrival_bonus):
        self.nw = network
        
        self.orig = origins # a numpy array of length = the number of agents
        self.dest = destinations # a numpy array of length = the number of agents
        self.qtt = quantity # a numpy array of length = the number of agents

        self.uc = np.array(self.nw.es["unit cost"])
        self.fc = np.array(self.nw.es["fixed cost"])
        self.cp = np.array(self.nw.es["capacity"])
        
        self.num_s = self.nw.vcount()
        self.num_e = self.nw.ecount()
        self.numagent = len(self.qtt)
        
        self.nw.add_vertex(attr='depot')
        self.nw.add_edge(self.num_s, self.num_s)

        self.state_space_dim = self.numagent + self.num_e
        self.action_space_dim = self.num_e
        
        self.arrival_bonus = arrival_bonus
        
    def action_available(self, state, agent_idx): # state = tuple // all agents' locations
        loc, cp, fc = state
        
        if loc[agent_idx] == self.num_s:
            return np.array([self.num_e])
        
        cp  = np.array(cp)
        incidents = self.nw.incident(loc[agent_idx], mode = "OUT") # edge-id
                
        action_list = np.ma.array(incidents, mask = cp[incidents]<self.qtt[agent_idx]).compressed()       
        
        return action_list
        
    def action_available2(self, state, agent_idx): #state = tensor // location of all agents
        state = state.numpy()
        loc = state[agent_idx]
        cp = state[self.numagent:self.numagent+self.num_e]
        
        if loc == self.num_s:
            return np.array([self.num_e])
        
        incidents = self.nw.incident(int(loc), mode = "OUT") #edge-id
                
        action_list = np.ma.array(incidents, mask = cp[incidents]<self.qtt[agent_idx]).compressed()       
        
        return action_list
    
    def step(self, state, action): # state = location of all agents + remaining cp + design variable for fc
        action_range = np.arange(self.numagent)
        loc, cp, fc = state
        loc_ = loc.copy()
        cp_ = cp.copy()
        fc_ = fc.copy()
        
        extended_cp = np.zeros(self.num_e+1)
        extended_cp[self.num_e] = math.inf
        extended_cp[:-1] = cp_
        
        extended_uc = np.zeros(self.num_e+1)
        extended_uc[:-1] = self.uc
        
        extended_fc = np.zeros(self.num_e+1)
        extended_fc[:-1] = fc_
        
        easycase = len(set(action)) == len(action) # every agent slects different edges
        
        if easycase:
            loc_ = np.array([self.nw.es[action[i]].target # next location
                             
                             if self.nw.es[action[i]].target != self.dest[i] # does it arrive at its destination node?
                             
                             else self.num_s # go to the waiting place
                             
                             for i in action_range])
            extended_cp[action] -= self.qtt
            extended_fc[action] = 1
            
            construction = extended_fc[:-1] - fc
            tmp5 = action == np.argwhere(construction)
            
            if len(tmp5) != 0:
                tmp6 = tmp5 * self.fc[np.argwhere(construction)] / np.vstack(np.sum(tmp5, axis = 1))
                fixed_cost = tmp6.sum(axis=0)
                
            else:
                fixed_cost = np.zeros(self.numagent)
            
            cost = extended_uc[action]*self.qtt + fixed_cost
        
        else:
            g = npi.group_by(action)
            unique = g.unique # same as set 
            idx = g.split_array_as_list(action_range)
                        
            ifdup = g.count>1
            ifuni = ~ifdup
            
            dup = unique[ifdup] # duplicated edge indices
            uni = unique[ifuni] # unique edge indices
            
            # grouping indicies of agents who choose the same edge
            idx_dup = np.asarray(idx)[ifdup]
            dup_group_num = len(idx_dup)
            cp_minus = np.array([self.qtt[idx_dup[i]].sum() for i in range(dup_group_num)]) # sum of demands of agents who choose the same edge
            extended_cp[dup] -= cp_minus
            
            # indices of agents whose chosen edge is not duplicated
            if len(uni) != 0:
                idx_uni = np.concatenate(np.asarray(idx)[ifuni])
                extended_cp[uni] -= self.qtt[idx_uni]
                
            neg_cp = np.nonzero(extended_cp<0) #edge number whose capacity is negative
            
            if len(neg_cp[0]) != 0:
                Back = np.in1d(action, neg_cp) # Agents who get bounced back = True
            
                tmp1 = np.in1d(dup, neg_cp[0]) # for example: [True, True, False] There are three groups and True = a group that results in a negative remaining capacity
                tmp2 = idx_dup[tmp1] # for example: array([array([1,8]), array([6, 7, 9])])
                tmp3 = np.concatenate(tmp2)
                
                pass_agents_list = []

                for i in range(len(neg_cp[0])): #neg_cp action 
                    
                    rc = cp[neg_cp[0][i]].copy()
                    random_num = np.random.choice(range(len(tmp2[i])))
                    pass_agent = tmp2[i][random_num]
                    rc -= self.qtt[pass_agent]
                    
                    x = self.qtt[np.delete(tmp2[i], random_num)] > rc
                    pass_agents = [pass_agent] 
                    remaining_agents = tmp2[i]
                    
                    while not np.all(x):
                        remaining_agents = np.delete(remaining_agents, random_num) # in terms of agent idx
                        x = self.qtt[remaining_agents] > rc
                        if ~np.all(x):
                            random_num = np.random.choice(np.arange(len(x))[~x])
                            pass_agent = remaining_agents[random_num]
                            pass_agents.append(pass_agent)
                            rc -= self.qtt[pass_agent]
                    
                    pass_agents_list.append(pass_agents) # agents - winner
                        
                Pass = np.concatenate(np.asarray(pass_agents_list))
                            
                tmp4 = tmp3[np.in1d(tmp3, Pass, invert = True)] # agents - loser
            
                Back[Pass] = False
            
                idx_move = np.where(~Back)[0] # index of agents who is allowed to go to the next nodes
                
                group_sum = [sum(self.qtt[pass_agents_list[i]]) for i in range(len(neg_cp[0]))]
                extended_cp[neg_cp] = cp[neg_cp] - group_sum
                
            else:
                idx_move = np.arange(self.numagent)
            
            
            
            loc_[idx_move] = np.array([self.nw.es[action[i]].target
                                      
                                      if self.nw.es[action[i]].target != self.dest[i]
                                      
                                      else self.num_s
                                      
                                      for i in idx_move])
    
            action = np.asarray(action)

            extended_fc[action[idx_move]] = 1
            
            construction = extended_fc[:-1] - fc
            tmp5 = action == np.argwhere(construction)
            
            if len(tmp5) != 0:
                
                if len(neg_cp[0]) != 0:
                    tmp5[:, tmp4] = False
                   
                tmp6 = tmp5 * self.fc[np.argwhere(construction)] / np.vstack(np.sum(tmp5, axis = 1))
                fixed_cost = tmp6.sum(axis=0)
            
            else:
                fixed_cost = np.zeros(self.numagent)
            
            shipping_cost = np.zeros(self.numagent)
            shipping_cost[idx_move] = extended_uc[action[idx_move]]*self.qtt[idx_move]
            cost = shipping_cost + fixed_cost
            
        next_state = (loc_, extended_cp[:-1], extended_fc[:-1])
        
        reward = -cost
        
        done = loc_ == self.num_s
        
        arrival_reward = [self.arrival_bonus if done[i] and action[i] != self.num_e else 0 for i in range(self.numagent)]
        reward += arrival_reward
        
#        if np.all(done):
#            print("Success!!!!!!!!!!")
#        else:
#            print(next_state[0])
 
        return next_state, reward, done
               
        
    def reset(self):
        state = (self.orig, self.cp, np.zeros(self.num_e))
        
        return state
    
#%%
def train(env, agents, memory, target_update, num_episodes, punishment):
    
    episode_rewards = []
    episode_success = []
    episode_length = []
    best_history_state = None
    best_history_action = None
    max_reward = -math.inf
    steps_done = 0
    
    for episode in range(num_episodes):
        
        state = env.reset()
        episode_reward = 0
        actions = np.zeros(env.numagent)
        done = np.zeros(env.numagent)
        step = 0
        
        history_state = [env.orig]
        history_action = []
        
        while not np.all(done):
            actions = [agents[i].get_action(state, steps_done) for i in range(env.numagent)]
            steps_done += 1
            next_state, reward, done = env.step(state, actions)
            
            history_state.append(next_state)
            history_action.append(actions)
            
            if not np.all(done):
                
                next_action_availability = [len(env.action_available(next_state, i)) for i in range(env.numagent)]
                no_actions = np.where(np.array(next_action_availability) == 0)[0]
                
                if len(no_actions) != 0 :    
                    done[no_actions] = True
                    reward[no_actions] -= punishment
                    
                    [memory[i].push(torch.from_numpy(np.concatenate((state[0], state[1]), axis = None)).float(), actions, torch.from_numpy(np.concatenate((next_state[0], next_state[1]), axis = None)).float(), reward, done[i]) if actions[i] != env.num_e else None for i in range(env.numagent)]
                    [agents[i].optimize_model() for i in range(env.numagent)]
                    
                    episode_reward += reward
                    next_loc = next_state[0]
                    success = len(next_loc[next_loc == env.num_s])

                    episode_rewards.append(episode_reward)
                    episode_success.append(success)
                    episode_length.append(step+1)
                    
                    if episode % 100 == 0:
                        print("Episode " + str(episode) + ": " + str(episode_reward.sum()))
                        print("step " + str(step+1))
                        print("No action available" + ": " + "agent # " + str(no_actions))
                        print("Best" + ": " + str(max_reward))
                    break
                    
            episode_reward += reward
            
            [memory[i].push(torch.from_numpy(np.concatenate((state[0], state[1]), axis = None)).float(), actions, torch.from_numpy(np.concatenate((next_state[0], next_state[1]), axis = None)).float(), reward, done[i]) if actions[i] != env.num_e else None for i in range(env.numagent)]            
            [agents[i].optimize_model() for i in range(env.numagent)]
            
            if np.all(done):
                episode_rewards.append(episode_reward)
                episode_success.append(env.numagent)
                episode_length.append(step+1)
                
                if episode_reward.sum() > max_reward:
                    max_reward = episode_reward.sum()
                    best_history_state = history_state
                    best_history_action = history_action
                
                if episode % 100 == 0:
                    print("Episode " + str(episode) + ": " + str(episode_reward.sum()))
                    print("step " + str(step+1))
                    print("Best" + ": " + str(max_reward))
                break
            
            state = next_state
            step += 1
            
            if episode % target_update == 0:
                [agents[i].target_net.load_state_dict(agents[i].policy_net.state_dict()) for i in range(env.numagent)]
        
    return (episode_rewards, episode_success, episode_length, best_history_state, best_history_action)

#%%  
def test(env, agents, c):
    state = env.reset()
    episode_reward = 0
    done = np.zeros(env.numagent)
    step = 0
    
    while not np.all(done):
        
        actions = [agents[i].get_greedy_action(state, c) for i in range(env.numagent)]
        if np.isnan(actions).sum() > 0:
            break
        next_state, reward, done = env.step(state, actions)
        episode_reward += reward
        
        if np.all(done):
#            print(episode_reward)            
#            print("step " + str(step))
            break
        
        state = next_state
#        print(state[0])
        step += 1
        
    return episode_reward
