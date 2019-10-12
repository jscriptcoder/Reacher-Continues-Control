import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .device import device

class Actor(nn.Module):
    def __init__(self, state_size, action_size, activ):
        super(Actor, self).__init__()
        
        self.input = nn.Linear(state_size, 64)
        self.hidden = nn.Linear(64, 64)
        self.output = nn.Linear(64, action_size)
        
        self.std = nn.Parameter(torch.zeros(action_size))
        
        self.activ = activ
        
        self.to(device)

    def forward(self, state, action=None):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.input(state)
        x = self.activ(x)
        
        x = self.hidden(x)
        x = self.activ(x)
        
        mean = torch.tanh(self.output(x))
        std = F.softplus(self.std)
        
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).mean(-1).unsqueeze(-1)
        entropy = dist.entropy().mean(-1).unsqueeze(-1)
        
        return action, log_prob, entropy