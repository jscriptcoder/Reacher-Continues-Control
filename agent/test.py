import numpy as np
from unityagents import UnityEnvironment

import torch
import torch.nn.functional as F
from torch.optim import Adam

from agent.a2c_agent import A2CAgent
from agent.ppo_agent import PPOAgent
from agent.config import Config
from agent.utils import EnvironmentAdapterForUnity

env = UnityEnvironment(file_name='./env/Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset()[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

unity_envs = EnvironmentAdapterForUnity(env, brain_name)
unity_envs.train_mode=False
config = Config()

config.num_agents = num_agents
config.envs = unity_envs

config.seed = 0
config.state_size = state_size
config.action_size = action_size
config.activ_actor = F.relu
config.lr_actor = 3e-4
config.hidden_actor = (512, 512)
config.optim_actor = Adam
config.activ_critic = F.relu
config.lr_critic = 3e-4
config.hidden_critic = (512, 512)
config.optim_critic = Adam

#agent = A2CAgent(config)
agent = PPOAgent(config)

#agent.policy.load_state_dict(torch.load('a2c_actor_checkpoint.ph', 
#                                        map_location='cpu'))
agent.policy.load_state_dict(torch.load('ppo_actor_checkpoint.ph', 
                                        map_location='cpu'))

agent.eval_episode(1)