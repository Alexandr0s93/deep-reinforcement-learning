import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-3         # learning rate of actor
LR_CRITIC = 1e-3        # learning rate of critic
TAU = 1e-3              # for soft update of target parameters
NOISE_DEC = 0.999       # decay rate for noise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): Dimension of each State
            action_size (int): Dimension of each Action
            seed (int): Random Seed
            lr_decay (float): Decay float for alpha learning rate
            DOUBLE DQN (boolean): Indicator for Double Deep Q-Network
            DUEL DQN (boolean): Indicator for Duel Deep Q-Network
            PRIORITISED_EXPERIENCE (boolean): Indicator for Prioritized Experience Replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        
        # Initialize Replay Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize Noise Process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = NOISE_DEC
        
    def act(self, state, noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(_device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()

        if noise:
            action += self.noise_decay * self.noise.sample()  # Add Noise for Exploration
            self.noise_decay *= self.noise_decay              # Decay Noise to balance Exploitation

        return np.clip(action, -1, 1)
    
    def step(self, states, actions, rewards, next_states):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        # Save experience in memory
        self.memory.add(states, actions, rewards, next_states)

        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > _batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        states, actions, rewards, next_states = experiences
        
        # ---------------------------- Update Critic ---------------------------- #
        actions_next = self.actor_target(next_states)                   # Get Next Actions from Actor Target
        Q_targets_next = self.critic_target(next_states, actions_next)  # Get Next-State Q-Values from Critic Target
        Q_targets = rewards + (_gamma * Q_targets_next)                 # Compute Target Q
        Q_expected = self.critic_local(states, actions)                 # Compute Expected Q
        critic_loss = F.mse_loss(Q_expected, Q_targets)                 # Compute Critic Loss
        
        # Minimize the Loss for Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- Update Actor ---------------------------- #
        
        actions_pred = self.actor_local(states)                         # Compute Next Actions from Actor Local
        actor_loss = -self.critic_local(states, actions_pred).mean()    # Get Actor Loss
        
        # Minimize the Loss for Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- Update Target Networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
     
    def soft_update(self, local_model, target_model, tau = TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)