# Report: Reacher using Policy Gradient

## Learning Algorithm

What we're dealing with here is an envirornment with continuous observation space that consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm, and continues action space, a vector with 4 numbers, corresponding to torque applicable to two joints. Policy Gradient methods are the right fit for continuos action space.

I'll try to solve this environment using two PG algorithms, using the latest Actor-Critic methods:

- Advantage Actor Critic or A2C. [A3C paper](https://arxiv.org/abs/1602.01783). Note A3C is the asyncronous version of A2C
- Proximal Policy Optimization. [Paper](https://arxiv.org/abs/1707.06347)

I'll be also computing advantages using λ-returns with Generalized Advantage Estimation or GAE. [Paper](https://arxiv.org/abs/1506.02438). Policy gradient, while unbiased, have high variance. This paper proposes ways to dramatically reduce variance, but this unfortunately comes at the cost of introducing bias. [Source](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)

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

## References
