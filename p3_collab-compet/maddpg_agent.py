from replay_buffer import ReplayBuffer
from ddpg_agent import DDPGAgent

import torch
import numpy as np

BUFFER_SIZE = int(1e5)  # Replay Buffer Size
BATCH_SIZE = 250        # Minibatch Size

class MADDPG:

    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize a Multi Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents used
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
        self.agents = [DDPGAgent(self.state_size, self.action_size, seed) for x in range(self.num_agents)]
        
        # Initialize Memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            for agent in self.agents:
                agent.learn(experiences)

    def act(self, states, add_noise=True, noise_factor=1.0):
        """Returns actions for given state as per current policy for each agent."""
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise, noise_factor)
        return actions

    def save_weights(self):
        """Saves weights of trained Agents."""
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'weights/agent{}_checkpoint_actor.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'weights/agent{}_checkpoint_critic.pth'.format(index+1))
    
    def reset(self):
        """Resets the Noise for all Agents."""
        for agent in self.agents:
            agent.noise.reset()