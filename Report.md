# Report: Reacher using Policy Gradient

## Learning Algorithm

What we're dealing with here is an envirornment with continuous observation space that consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm, and continues action space, a vector with 4 numbers, corresponding to torque applicable to two joints. Policy Gradient methods are the right fit for continuos action space.

I'll try to solve this environment using two PG algorithms, using the latest Actor-Critic methods:

Proximal Policy Optimization or PPO, which will be built on top of A2C. Paper
I'll be also computing returns using Generalized Advantage Estimation or GAE. Paper

- [Advantage Actor Critic or A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f). [A3C paper](https://arxiv.org/abs/1602.01783). Note A3C is the asyncronous version of A2C
- [Proximal Policy Optimization](https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12). [Paper](https://arxiv.org/abs/1707.06347)

### Hyperparameters


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

