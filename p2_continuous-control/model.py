import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy based method) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Define Actor Layers
        self.bn1d = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_size)
        
        # Reset Weights & Bias
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the parameters of the hidden units."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)        
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.bn1d(state)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        x = torch.tanh(x)       
        return x


class Critic(nn.Module):
    """Critic (Value based method) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_layers (list): Number of nodes in hidden layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Define Critic Layers
        self.bn1d = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.output = nn.Linear(128, 1)
        
        # Reset Weights & Bias
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the parameters of the hidden units."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.bn1d(state)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        return x