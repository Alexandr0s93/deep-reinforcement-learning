import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Actor (Policy) Model using a Single DQN."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Define Deep Q-Network Layers
        self.dqn_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        q_values = self.dqn_layers(state)
        return q_values
    

class DuelQNetwork(nn.Module):
    """Actor (Policy) Model using a Duel DQN."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Define Feature Layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Define Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(32, 1)
        )
        # Define Advantage Layers
        self.advantage_stream = nn.Sequential(
            nn.Linear(32, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.feature_layers(state)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        q_values = values + (advantages - advantages.mean())
        return q_values