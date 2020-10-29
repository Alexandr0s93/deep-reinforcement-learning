from collections import namedtuple, deque
import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = 0., beta = 1., beta_anneal = 1.):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): parameter to control magnitude of prioritization
            beta (float): parameter to copensate for the non-uniform probabilities
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = np.empty(self.buffer_size, dtype=[
            ("state", np.ndarray),
            ("action", np.int),
            ("reward", np.float),
            ("next_state", np.ndarray),
            ("done", np.bool),
            ('prob', np.double)])
        self.seed = random.seed(seed)
        
        # Memory specific containers
        self.memory_idx = 0             
        self.memory_sample = np.empty(self.batch_size)
        self.memory_sample_idx = np.empty(self.batch_size)
        
        # Prioritized Experience Replay Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        
        self.max_p = 0.1             # Max Probability to choose
        self.min_p = 0.00001            # Constant e for non-zero Probability
        
        self.p = np.empty(buffer_size)  # Probabilities array
        self.w = np.empty(buffer_size)  # Weights array
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory[self.memory_idx]['state'] = state
        self.memory[self.memory_idx]['action'] = action
        self.memory[self.memory_idx]['reward'] = reward
        self.memory[self.memory_idx]['next_state'] = next_state
        self.memory[self.memory_idx]['done'] = done
        self.memory[self.memory_idx]['prob'] = self.max_p
        
        self.memory_idx = (self.memory_idx + 1) % self.buffer_size  # Restart Memory Index if buffer is full
    
    def sample(self):
        """Sample a batch based on prioritized experiences from memory."""
        self.p = self.memory['prob'] / self.memory['prob'].sum()
        self.memory_sample_idx = np.random.choice(self.buffer_size, self.batch_size, replace=False, p=self.p)
        self.memory_sample = self.memory[self.memory_sample_idx]
        
        self.w = np.multiply(self.memory['prob'], self.buffer_size)   # Weights Denominator
        self.w = np.power(self.w, -self.beta, where=self.w!=0)        # Calculate Weights
        self.w = self.w/self.w.max()                                  # Normalize Weights
        
        states = torch.from_numpy(np.vstack(self.memory_sample['state'])).float().to(device)
        actions = torch.from_numpy(np.vstack(self.memory_sample['action'])).long().to(device)
        rewards = torch.from_numpy(np.vstack(self.memory_sample['reward'])).float().to(device)
        next_states = torch.from_numpy(np.vstack(self.memory_sample['next_state'])).float().to(device)
        dones = torch.from_numpy(np.vstack(self.memory_sample['done']).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(self.w[self.memory_sample_idx]).float().to(device)
        
        return (states, actions, rewards, next_states, dones, weights)
    
    def update_probabilities(self, td_error):
        """Update probabilities based on TD Error."""
        self.beta = min(1, self.beta*self.beta_anneal)             # Anneal Beta hyperparameter
        
        td_error = abs(td_error)**self.alpha + self.min_p          # Calculate absoluste delta
        
        self.memory_sample['prob'] = td_error
        self.memory[self.memory_sample_idx] = self.memory_sample   # Update Probabilities in memory
        
        self.max_p = self.memory_sample['prob'].max()              # Update Max Probability

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory[self.memory['prob'] > 0])