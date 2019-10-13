import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .device import device

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activ):
        super().__init__()
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
    
        self.std = nn.Parameter(torch.zeros(action_size))
        
        self.activ = activ
        
        self.to(device)

    def forward(self, state, action=None):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))

        mean = torch.tanh(self.layers[-1](x))
        std = F.softplus(self.std)
        
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        
        return action, log_prob, entropy