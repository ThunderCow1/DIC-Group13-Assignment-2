import random
import datetime
import pygame
import numpy as np
import math
from collections import defaultdict
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from robot import Robot
import torch 
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)  # output_dim = number of discrete actions
        )

    def forward(self, x):
        return self.model(x)  


class PPOAgent:
    def __init__(self, robot: Robot):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot = robot
        self.actions = ['turn_right', 'turn_left', 'accelerate', 'break', 'break_hard']
        self.state_dim = 7 + 5  # 7 state vars + 5 distance sensors
        self.actor = MLPNetwork(self.state_dim, len(self.actions)).to(self.device)
        self.critic = MLPNetwork(self.state_dim, 1).to(self.device)
        self.memory = []  
        self.rewards = []
        self.dones = []
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def _build_state(self, x, y, orientation, speed, target_x, target_y, angle_diff, distances):
        state = np.array([x, y, orientation, speed, target_x, target_y, angle_diff] + distances, dtype=np.float32)
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def select_action(self, x, y, orientation, speed, target_x, target_y, angle_diff, distances):
        state = self._build_state(x, y, orientation, speed, target_x, target_y, angle_diff, distances)

        with torch.no_grad():
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

        action = self.actions[action_idx.item()]
        value = self.critic(state)

        self.memory.append((state, action_idx, log_prob, value))
        return [action]  # Wrap in list to maintain compatibility with Robot

    def _update(self, reward, old_pos, action_list):
        self.rewards.append(reward)
        self.dones.append(False)

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        rewards = self.rewards
        values = [v.item() for (_, _, _, v) in self.memory]
        values.append(0)

        returns = []
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return returns, advantages

    def train(self, epochs=4, clip_eps=0.2):
        returns, advantages = self.compute_returns_and_advantages()

        states, actions, old_log_probs, _ = zip(*self.memory)
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(old_log_probs)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(epochs):
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO ratio
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = nn.functional.mse_loss(values, returns)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.memory = []
        self.rewards = []
        self.dones = []


