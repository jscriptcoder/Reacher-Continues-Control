import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from .actor import Actor
from .critic import Critic
from .policy import Policy
from .device import device

class A2CAgent:
    def __init__(self, config):
        self.config = config
        
        self.policy = Policy(config.state_size, config.action_size)

#        self.policy = Actor(config.state_size, 
#                            config.action_size, 
#                            config.activ_actor)
#        
#        self.value = Critic(config.state_size, 
#                            config.activ_critic)
        
        self.optim = config.optim(self.policy.parameters(), lr=config.lr)
#        
#        self.optim_policy = config.optim_actor(self.policy.parameters(), 
#                                               lr=config.lr_actor)
#        self.optim_value = config.optim_critic(self.value.parameters(), 
#                                               lr=config.lr_critic)
        
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        self.reset()
    
    def reset(self):
        self.state = self.config.envs.reset()
        self.total_steps = 0
        self.all_done = False
    
    def act(self, state):
        self.policy.eval()
        with torch.no_grad():
            action, _, _, _= self.policy(state)
#            action, _, _= self.policy(state)
        self.policy.train()
        
        return action
    
    def collect_data(self):
        steps = self.config.steps
        envs = self.config.envs
        state = self.state
        
        states = []
        actions = []
        log_probs = []
        values = []
        entropies = []
        rewards = []
        masks = []
                
        for _ in range(steps):
            state = torch.FloatTensor(state).to(device)
            
            action, log_prob, entropy, value = self.policy(state)
#            action, log_prob, entropy = self.policy(state)
#            value = self.value(state)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            done = np.array(done)
            
            rewards.append(torch.FloatTensor(reward).unsqueeze(-1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1).to(device))
            
            state = next_state
            
            self.total_steps += 1
            
            if done.any():
                self.all_done = True
                break
        
        self.state = state
        _, _, _, next_value = self.policy(state)
#        next_value = self.value(state)
        
        return (log_probs, 
                entropies, 
                values, 
                rewards, 
                masks, 
                states, 
                actions, 
                next_value)
    
    def compute_return(self, values, rewards, masks, next_value):
        num_agents = self.config.num_agents
        use_gae = self.config.use_gae
        gamma = self.config.gamma
        lamda = self.config.lamda
        
        values = values + [next_value]
        returns = []
        
        R = next_value.detach()
        GAE = torch.zeros((num_agents, 1))
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            
            if use_gae:
                value = values[i].detach()
                value_next = values[i+1].detach()
                
                # δ = r + γV(s') - V(s)
                delta = reward + gamma * value_next * masks[i] - value
                
                # GAE = δ' + λγδ
                GAE = delta + lamda * gamma * GAE * masks[i]
                
                returns.insert(0, GAE + value)
            else:
                # R = r + γV(s')
                R = reward + gamma * R * masks[i]
                
                returns.insert(0, R)
        
        return returns
    
    def update(self, policy_loss, value_loss=None):
        grad_clip = self.config.grad_clip
#        grad_clip_actor = self.config.grad_clip_actor
#        grad_clip_critic = self.config.grad_clip_critic
        
        self.optim.zero_grad()
        
        policy_loss.backward()
        
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
        
        self.optim.step()
        
#        self.optim_policy.zero_grad()
#        policy_loss.backward()
#        
#        if grad_clip_actor is not None:
#            nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip_actor)
#        
#        self.optim_policy.step()
#        
#        self.optim_value.zero_grad()
#        value_loss.backward()
#        
#        if grad_clip_critic is not None:
#            nn.utils.clip_grad_norm_(self.value.parameters(), grad_clip_critic)
#        
#        self.optim_value.step()
    
    def learn(self, 
              log_probs, 
              values, 
              returns, 
              **kwargs):
        
        entropies = kwargs['entropies']
        
        ent_weight = self.config.ent_weight
        val_loss_weight = self.config.val_loss_weight
        
        # List to torch.Tensor
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)
        returns = torch.cat(returns)
        
        # A(s, a) = r + γV(s') - V(s)
        advantages = returns - values.detach()
        
        policy_loss = (-log_probs * advantages - ent_weight * entropies).mean()
        value_loss = val_loss_weight * (returns - values).pow(2).mean()
        
        self.update(policy_loss + value_loss)
        
        return policy_loss, value_loss
    
    def step(self):
        
        (log_probs, 
         entropies, 
         values, 
         rewards, 
         masks, 
         states, 
         actions, 
         next_value) = self.collect_data()
        
        returns = self.compute_return(values, rewards, masks, next_value)

        policy_loss, value_loss = self.learn(log_probs, 
                                             values, 
                                             returns, 
                                             entropies=entropies,
                                             states=states, 
                                             actions=actions)
        
        return rewards, policy_loss, value_loss
    
    def train(self):
        num_episodes = self.config.num_episodes
        env_solved = self.config.env_solved
        size_score = self.config.size_score
        envs = self.config.envs
        
        scores = []
        scores_window = deque(maxlen=size_score)
        best_mean_score = -np.inf
        
        for i_episode in range(1, num_episodes+1):
            self.reset()
            score = 0
            
            while True:
                rewards, policy_loss, value_loss = self.step()
                
                score += torch.cat(rewards).cpu().numpy().mean()
                
                if self.all_done:
                    break
            
            scores.append(score)
            scores_window.append(score)
            mean_score = np.mean(scores_window)
            
            print('\rEpisode {}\tPolicy loss: {:.3f}\tValue loss: {:.3f}\tAvg Score: {:.3f}'\
                  .format(i_episode, 
                          policy_loss, 
                          value_loss, 
                          mean_score), end='')
            
            if i_episode % size_score == 0:
                
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    print('\r* Best score so far: {:.3f}'.format(mean_score))
                
                if mean_score >= env_solved:
                    print('Environment solved with {:.3f}!'.format(mean_score))
                    break;
        
        envs.close()
    
    def run_episode(self, debug=True):
        env = self.config.envs
        state = env.reset()
        
        total_score = 0
        
        env.render()
        
        while True:
            action = self.act(state)
            state, reward, done, _ = env.step(action.cpu().numpy())
            
            env.render()
            
            avg_reward = np.mean(reward)
            total_score += avg_reward
            
            if debug:
                print('Avg reward: {:.2f}'.format(avg_reward))

            if np.array(done).all():
                break
        
        print('Total reward: {:.2f}'.format(total_score))
                
        env.close()
    
    def summary(self, agent_name='A2C Agent'):
        print('{}:'.format(agent_name))
        print('==========')
        print('')
        print('Policy Network:')
        print('---------------')
        print(self.policy)
#        print('')
#        print('Value Network:')
#        print('--------------')
#        print(self.value)