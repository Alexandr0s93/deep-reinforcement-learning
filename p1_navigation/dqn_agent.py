import numpy as np
import random

from model import QNetwork, DuelQNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 1e-3               # learning rate 
UPDATE_EVERY = 16       # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, lr_decay=0.9999,
                 double_dqn=False, duel_dqn=False, prio_exp=False):
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
        self.lr_decay = lr_decay
        self.DOUBLE_DQN = double_dqn
        self.DUEL_DQN = duel_dqn
        self.PRIORITISED_EXPERIENCE = prio_exp

        # Determine Deep Q-Network for use
        if self.DUEL_DQN:
            self.qnetwork_local = DuelQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        # Initialize Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Determine if Prioritized Experience will be used
        if self.PRIORITISED_EXPERIENCE:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, 
                                                  alpha = 0.6, beta = 0.4, beta_anneal = 1.0001)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.PRIORITISED_EXPERIENCE:
            states, actions, rewards, next_states, dones, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        
        if self.DOUBLE_DQN:
            # Select max Action for Next State from Local NN
            max_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            # Evaluate max Action with Target NN
            Q_targets_next = self.qnetwork_target(next_states).gather(1, max_action)
        else:
            # Get Max Predicted Q values for next state from Target NN
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Predicted Q values for current state
        Q_targets = rewards + (gamma*Q_targets_next*(1 - dones))
        
        # Get Expected Q values from Local NN
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        if self.PRIORITISED_EXPERIENCE:
            td_error = (Q_expected - Q_targets).squeeze_()           # Compute TD Error
            td_error_detached = td_error.detach()
            
            self.memory.update_probabilities(td_error_detached)      # Update Probabilities
            
            loss = ((td_error**2)*weights).mean()                    # Compute Weighted Loss
        else:
            loss = F.mse_loss(Q_expected, Q_targets)                 # Compute Loss

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- Update Target Network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
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