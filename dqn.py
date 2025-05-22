import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import math
import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info

class DQN(nn.Module):

    def __init__(self, n_actions):
        """
        Initializes the Deep Q-Network (DQN) model.

        Args:
            n_actions (int): The number of possible actions the agent can take (i.e., output size of the network).
        """
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the DQN model.
    
        Args:
            x (torch.Tensor): Input tensor representing the state (typically an image),
                              expected shape is [batch_size, channels, height, width].
    
        Returns:
            torch.Tensor: Output tensor containing Q-values for each action.
        """
        x = x / 255.0
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x

class ReplayBuffer:
    
    def __init__(self, capacity, state_dim):
        """
        Initializes the replay buffer.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            state_dim (int or tuple): Shape of a single state.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the replay buffer and processes them into tensors.
    
        Args:
            batch_size (int): Number of transitions to sample from the buffer.
            device (torch.device): Device on which to place the tensors (e.g., 'cpu' or 'cuda').
    
        Returns:
            Tuple[torch.Tensor]: A tuple of tensors: (states, actions, rewards, next_states, dones),
                                 all moved to the specified device.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.tensor(state) for state in states]).float().to(device)
        next_states = torch.stack([torch.tensor(next_state) for next_state in next_states]).float().to(device)
        
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        
        return len(self.buffer)


class DQNAgent:
    
    def __init__(self, state_dim, action_dim, buffer, ddqn, gamma=0.99, epsilon=0.8, epsilon_start=0.8, epsilon_min=0.1, batch_size=64, learning_rate=1e-4):
        """
        Initializes the DQN agent with necessary parameters and networks.
    
        Args:
            state_dim (int or tuple): Dimension or shape of the input state.
            action_dim (int): Number of possible actions in the environment.
            buffer (ReplayBuffer): Experience replay buffer for sampling training data.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_start (float): Starting value for epsilon in decay.
            epsilon_min (float): Minimum value for epsilon after decay.
            batch_size (int): Number of transitions sampled per training step.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.policy_net = DQN(action_dim).to(device)
        self.target_net = DQN(action_dim).to(device)
        self.ddqn = ddqn
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        
        self.batch_size = batch_size
        self.buffer = buffer

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.TAU = 0.005


    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            action = torch.argmax(q_values, dim=1).item()
            return action
            
    def decay(self, episode, factor):
        """
        Applies exponential decay to the epsilon value based on the current episode.
    
        Args:
            episode (int): The current episode number.
            factor (float): Decay rate controlling how quickly epsilon approaches epsilon_min.
        """
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-factor * episode)

    def train(self):
        """Train the DQN agent."""
        if len(self.buffer) < self.batch_size:
            return  # Not enough data to train

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Compute Q-values for current states
        state_values = self.policy_net(states).gather(1, actions) # Select Q-values for the taken actions

        # Compute Q-values for next states using the target model
        # If ddqn is set to TRUE, calculate next_q_values by DDQN logic, else calculate them by DQN logic
        with torch.no_grad():
            if self.ddqn:
                next_actions = self.policy_net(next_states).max(dim=1, keepdim=True)[1]
                next_q_values = self.target_net(next_states).gather(1, next_actions)            
            else:
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]

        # Compute the target Q-value
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(state_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Soft update of the Target Network
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            
        self.target_net.load_state_dict(target_net_state_dict)

    def policy(self, state):
        """
        Selects the best action for a given state using the current policy network.
        Optimized for inference by disabling gradient computation.
    
        Args:
            state (np.ndarray or torch.Tensor): The current state, expected shape [H, W, C].
    
        Returns:
            int: Index of the action with the highest Q-value.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        
        with torch.no_grad():
            q_values = self.policy_net(state)
    
        action = torch.argmax(q_values, dim=1).item()
        return action

    def save_full_model(self, filename):
        """
        Save the entire model (architecture and weights) to a file.
    
        Args:
            filename (str): The path where the model will be saved.
        """
        torch.save(self.policy_net, filename)
        
        print(f"Full model saved to {filename}")

    def load_full_model(self, filename):
        """
        Load the entire model (architecture and weights) from a file.
    
        Args:
            filename (str): The path from which the model will be loaded.
        """
        model = torch.load(filename, weights_only=False)
        self.policy_net = model.to(device)
        self.target_net = copy.deepcopy(model).to(device)
        
        print(f"Full model loaded from {filename}")





        