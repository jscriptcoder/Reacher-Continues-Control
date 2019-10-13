import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .device import device

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.base = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
        )
        
        self.mean = nn.Sequential(
            nn.Linear(64, action_size),
            nn.Tanh(),
        )
        
        self.std = nn.Parameter(torch.ones(1, action_size))
        
        self.value = nn.Linear(64, 1)
        
        self.to(device)

    def forward(self, state, action=None):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        base_out = self.base(state)
        
        mean = self.mean(base_out)
        std = self.std(base_out)
        
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        
        value = self.value(base_out)
        
        return action, log_prob, entropy, value