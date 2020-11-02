# Continuous-Control Report

### Introduction

The project is considered solvable if the Agents achieve a **Max 0.5 Average Score** over the last 100-episodes.

The following implementation achieves this goal in **???** by utilizing **2 Agents** that use the `Deep Deterministic Policy Gradient (DDPG) Algorithm`. In this approach the below components are used:

- Actor Neural Network
- Critic Neural Network
- Replay Buffer
- Ornstein-Uhlenbeck Process

### Actor NN

The Actor NN is a **Policy based method** where Actions are mapped directly on input states. The Actor NN uses the following Architecture:

- Batch Normalization Layer: 33 Units
- Hidden Layer: 33 Units
- Hidden Layer: 128 Units
- Output Layer: 128 Units

~~~python
Actor(
  (fc1): Linear(in_features=24, out_features=200, bias=True)
  (fc2): Linear(in_features=200, out_features=150, bias=True)
  (output): Linear(in_features=150, out_features=2, bias=True)
)
~~~

For every Linear Layer, a ReLU activation function follows before proceeding to the next Layer. At the end a Tanh activation is happening on the output layer.

### Critic NN

The Critic NN is a **Value based method** where Q-values are being calculated for each state. The Critic NN uses the following Architecture:

- Batch Normalization Layer: 33 Units
- Hidden Layer: 33 Units
- Hidden Layer: 132 Units
- Output Layer: 128 Units

~~~python
Critic(
  (fc1): Linear(in_features=52, out_features=200, bias=True)
  (fc2): Linear(in_features=200, out_features=150, bias=True)
  (output): Linear(in_features=150, out_features=1, bias=True)
)
~~~

For every Linear Layer, a ReLU activation function follows before proceeding to the next Layer.

### Replay Buffer

During training, the Agent pick a batch of past experiences **at random** to learn and update its weights. In that way the Agent is not learning sequentially from experiences and the correlation among examples is avoided.

### Ornstein-Uhlenbeck Process

To facilitate **Exploration** a Noise component is added which decays over time in order to balance **Exploitation** during training.

### DDPG Algorithm

The DDPG Algorithm utilizes the Actor Network to choose an action based on a particular state and then uses the Critic network to evaluate that action and learn. The overall Agents' scores can be seen in the below graph followed by the individual Agent's scores:

<p align="center">
  <img src="../img/Overall_Agent_scores.png" />
</p>

As illustrated the Agents achieves an average score of 30 in 96 Episodes. During training of the Agent the following `Hyperparameters` were used.

#### 1. Environment Hyperparameters

- No Noise Decay has been used

#### 2. Agent Hyperparameters

- Buffer Size: int(1e5)
- Batch Size: 250
- Gamma: 0.99
- Tau: 1e-3
- LR_Actor: 1e-4
- LR_Critic: 1e-3

### Idea for Future Work

- Hyperparameters have been tuned to an extend but through further optimization the learning might be faster and the Agents could reach the environment's goal earlier.

- The implementation is based on a 2-Agents that use a DDPG algorithm. It would be interesting to see how this would compare with using other algorithms as well e.g. PP0.

- Apart from the Tennis environment there is another Challenge that uses the **Soccer** environment. It would be quite interesting to test the above implementation on this environment as well.

### References
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
