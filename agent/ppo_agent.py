import numpy as np
import torch
import torch.nn.functional as F

from .a2c_agent import A2CAgent

class PPOAgent(A2CAgent):
    def __init__(self, config):
        super().__init__(config)
    
    def random_indices(self, len_states, shuffle=True):
        ppo_batch_size = self.config.ppo_batch_size
        indices = np.arange(len_states)
        
        if shuffle:
            indices = np.random.permutation(indices)
        
        len_total = len_states // ppo_batch_size * ppo_batch_size
        batches = indices[:len_total].reshape(-1, ppo_batch_size)
        
        for batch in batches:
            yield batch
        
        rest = len_states % ppo_batch_size
        if rest > 0:
            yield indices[-rest:]
        
        
    
    def learn(self, 
              log_probs, 
              values, 
              returns, 
              **kwargs):
        
        states = kwargs['states']
        actions = kwargs['actions']
        
        ent_weight = self.config.ent_weight
        val_loss_weight = self.config.val_loss_weight
        ppo_clip = self.config.ppo_clip
        ppo_epochs = self.config.ppo_epochs
        
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        returns = torch.cat(returns)
        states = torch.cat(states)
        actions = torch.cat(actions).detach()
        
        advantages = returns - values
        
        for _ in range(ppo_epochs):
            sampler = self.random_indices(len(states))
            
            for batch_idx in sampler:
                sampled_state = states[batch_idx, :]
                sampled_action = actions[batch_idx]
                sampled_log_prob = log_probs[batch_idx, :]
                sampled_advantage = advantages[batch_idx, :]
                sampled_return = returns[batch_idx, :]

                (_, 
                 new_log_prob, 
                 entropy, 
                 new_value) = self.policy(sampled_state, sampled_action)
                
#                _, new_log_prob, entropy = self.policy(sampled_state, sampled_action)
#                new_value = self.value(sampled_state)
                
                ratio = (new_log_prob - sampled_log_prob.detach()).exp()
                
                surrogate_ratio = ratio * sampled_advantage
                surrogate_clip = ratio.clamp(1.0 - ppo_clip, 
                                             1.0 + ppo_clip) * sampled_advantage
                
                min_surrogate = torch.min(surrogate_ratio, surrogate_clip)
                
                policy_loss = (-min_surrogate - ent_weight * entropy).mean()
                value_loss = val_loss_weight * (sampled_return - new_value).pow(2).mean()
                
                self.update(policy_loss + value_loss)
#                self.update(policy_loss, value_loss)
        
        return policy_loss, value_loss
    
    def summary(self):
        super().summary('PPO Agent')