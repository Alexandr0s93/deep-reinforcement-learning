from model import Actor, Critic
from noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    
    def __init__(self, state_size, action_size, seed):
        """Initialize a Multi Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Actor
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        # Initialize Critic
        self.critic_local = Critic(state_size, action_size , seed).to(device)
        self.critic_target = Critic(state_size, action_size , seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Initialize Noise
        self.noise = OUNoise(action_size, seed)       

    def act(self, state, add_noise=True, noise_factor=1.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += noise_factor*self.noise.sample().reshape((-1, 2))
        return np.clip(action, -1, 1)

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
                    
        next_states_tensor = torch.cat(next_states, dim=1).to(device)
        states_tensor = torch.cat(states, dim=1).to(device)
        actions_tensor = torch.cat(actions, dim=1).to(device)
        
        # ---------------------------- Update Critic ---------------------------- #
        next_actions = [self.actor_target(states) for states in states]        
        next_actions_tensor = torch.cat(next_actions, dim=1).to(device)              # Get Next-state Actions from Target
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor) # Get Next-state Q-values from Target     
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))                 # Compute Q Target for State
        Q_expected = self.critic_local(states_tensor, actions_tensor)                # Compute Q Expected for State
        critic_loss = F.mse_loss(Q_expected, Q_targets)                              # Compute Critic Loss
        
        # Minimize the Loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #
        actions_pred = [self.actor_local(states) for states in states]        
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)              # Get Next-state Actions from Locals
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()   # Compute Actor Loss     
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()        
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)