import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size=30, hidden_size=128, output_size=40):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state):
        return self.network(state)

class ReplayMemory:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory.append((state, action, reward, next_state, done, next_valid_actions))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_function = nn.SmoothL1Loss()
        self.memory = ReplayMemory()
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
        self.steps = 0
        self.losses = []

    def select_action(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("No valid actions")

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)[0]

            action = max(
                valid_actions,
                key=lambda a: q_values[a].item()
            )
        return action

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory.push(state, action, reward, next_state, done, next_valid_actions)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
    
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions = zip(*batch)
    
        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )
    
        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )
    
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )
    
        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )
    
        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )
    
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        action_mask = torch.full((self.batch_size, 40), float("-inf"), device=self.device)
    
        for i, valid in enumerate(next_valid_actions):
            if dones[i].item():
                action_mask[i] = 0
            elif valid:
                action_mask[i, valid] = 0
            else:
                action_mask[i] = 0
    
        with torch.no_grad():
            online_q = self.model(next_states)
            next_actions = (online_q + action_mask).argmax(dim=1)
            
            target_q_values = self.target_model(next_states)
            next_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = (rewards + self.gamma * next_q * (1 - dones))
    
        loss = self.loss_function(current_q, target_q)
        self.optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
    
        self.optimizer.step()
        self.steps += 1
        self.losses.append(loss.item())
    
        if self.steps % 500 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
        return loss.item()

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "target": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps
        }, path)

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]