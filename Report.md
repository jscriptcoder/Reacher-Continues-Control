# Report: Reacher using Policy Gradient

## Learning Algorithm

What we're dealing with here is an envirornment with continuous observation space that consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm, and continues action space, a vector with 4 numbers, corresponding to torque applicable to two joints. Policy Gradient methods are the right fit for continuos action space.

I'll try to solve this environment using two PG algorithms, using the latest Actor-Critic methods:

- Advantage Actor Critic or A2C. [A3C paper](https://arxiv.org/abs/1602.01783). Note A3C is the asyncronous version of A2C
- Proximal Policy Optimization. [Paper](https://arxiv.org/abs/1707.06347)

I'll be also computing advantages using λ-returns with Generalized Advantage Estimation or GAE. [Paper](https://arxiv.org/abs/1506.02438). Policy gradient, while unbiased, have high variance. This paper proposes ways to dramatically reduce variance, but this unfortunately comes at the cost of introducing bias. [Source](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)

### Hyperparameters

Following is a list of all the hyperparameters used and their values:

#### General params
- ```seed = 0```
- ```num_agents = 20```
- ```num_episodes = 1000```
- ```steps = 2000```, steps done per episode, until ```done=True```
- ```state_size = 33```
- ```action_size = 4```, vector of 4 values ranging between -1 and 1
- ```gamma = 0.99```, discount factor
- ```ent_weight = 0.01```, entropy coefficient for exploration
- ```val_loss_weight = 1```. This weight makes sense when using only one network with two heads for actor and critic, controlling how much weight the value loss has over the combined loss. Since I'm using two separate networks its value should be 1

We're gonna evaluate the environment after each episode just once. When we reach ```env_solved``` avarage score, then we'll run a full evaluation, that means, we're gonna evaluate ```times_solved``` times (this is required to solve the env) and avarage all the rewards
- ```env_solved = 30```
- ```times_solved = 100```

#### Actor (Policy network) params
- ```activ_actor = F.relu```
- ```lr_actor = 3e-4```
- ```hidden_actor = (512, 512)```, two hidden layers
- ```optim_actor = Adam```
- ```grad_clip_actor = 5```

#### Critic (Value network) params
- ```activ_critic = F.relu```
- ```lr_critic = 3e-4```
- ```hidden_critic = (512, 512)```, two hidden layers
- ```optim_critic = Adam```
- ```grad_clip_critic = 5```

#### PPO hyperparams
- ```ppo_clip = 0.2```
- ```ppo_epochs = 10```. Controls how many times we're gonna update the policy using mini-batches of previously collected trajectories
- ```ppo_batch_size = 32```

#### GAE
- ```use_gae = True```
- ```lamda = 0.95```, parameter to compute GAE

### Algorithms
1. **Advantage Actor Critic or A2C**:

2. **Proximal Policy Optimization or PPO**:

### Neural Networks Architecture

1. **A2C**
<img src="images/vanilla_dqn_agent.png" width="450" title="Vanilla DQN Agent" />

2. **PPO**
<img src="images/double_dqn_agent.png" width="450" title="Double DQN Agent" />

## Plot of Rewards

1. **A2C**:

2. **PPO**:

## Ideas for Future Work

## References
