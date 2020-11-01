from collections import namedtuple, deque
import copy
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-4         # learning rate of actor
LR_CRITIC = 1e-3        # learning rate of critic
TAU = 1e-3              # for soft update of target parameters
LEARN_EVERY = 2         # learn every LEARN_EVERY steps
LEARN_NB = 1            # how often to execute the learn-function each LEARN_EVERY steps

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
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0
        self.i_learn = 0  # for learning every n steps
        
        # Actor Network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        
        # Initialize Replay Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize Noise Process
        self.noise = OUNoise(20*action_size, seed)
            
    def act(self, state, noise=True, noise_factor = 1.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()

        if noise:
            action += noise_factor * self.noise.sample().reshape((-1, 4))  # Add Noise for Exploration

        return np.clip(action, -1, 1)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        # Save experience in memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        self.i_learn = (self.i_learn + 1) % LEARN_EVERY
        # Learn every LEARN_EVERY steps if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.i_learn == 0:
            for _ in range(LEARN_NB):
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
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- Update Critic ---------------------------- #
        actions_next = self.actor_target(next_states)                   # Get Next Actions from Actor Target
        Q_targets_next = self.critic_target(next_states, actions_next)  # Get Next-State Q-Values from Critic Target
        Q_targets = rewards + (GAMMA*Q_targets_next*(1 - dones))        # Compute Target Q
        Q_expected = self.critic_local(states, actions)                 # Compute Expected Q
        critic_loss = F.mse_loss(Q_expected, Q_targets)                 # Compute Critic Loss
        
        # Minimize the Loss for Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
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