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

#change to ReLu


# slightly deeper network 
class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.model(x)


class PPOAgent:
    def __init__(self, robot: Robot, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, lam=0.95, clip_eps=0.2, entropy_coef=0.02):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot = robot
        
        #PPO hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        #self.actions = ['turn_right', 'turn_left', 'accelerate', 'break', 'break_hard']
        self.actions = [
            'noop',                # do nothing
            'accelerate',
            'brake',
            'turn_left',
            'turn_right',
            'accel_left',
            'accel_right',
            'brake_left',
            'brake_right'
        ]

        #Network dimensions
        self.state_dim = None
        self.actor = None
        self.critic = None
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_dones = []
        
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

    def _build_state(self, x, y, orientation, speed, target_x, target_y, angle_diff, distances):
        #5 distance sensor readings
        if len(distances) != 5:
            distances = distances[:5]
            while len(distances) < 5:
                distances.append(256.0)
        
        # Normalize all state components to reasonable ranges
        state_list = [
            x / 1000.0,
            y / 1000.0,
            orientation / 360.0,
            speed / 10.0,
            target_x / 1000.0,
            target_y / 1000.0,
            angle_diff / 180.0
        ]
        
        for d in distances:
            state_list.append(d / 300.0)
        
        if self.state_dim is None:
            self.state_dim = len(state_list)
            self._initialize_networks()
        
        state = np.array(state_list, dtype=np.float32)
        state = np.clip(state, -5.0, 5.0)
        
        # Handle invalid values (NaN or infinity)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            state = np.zeros_like(state)
        
        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def _initialize_networks(self):
        self.actor = MLPNetwork(self.state_dim, len(self.actions), hidden_dim=128).to(self.device)
        self.critic = MLPNetwork(self.state_dim, 1, hidden_dim=128).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, x, y, orientation, speed, target_x, target_y, angle_diff, distances):
        state = self._build_state(x, y, orientation, speed, target_x, target_y, angle_diff, distances)

        if self.actor is None:
            return []

        with torch.no_grad():
            logits = self.actor(state)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = self.critic(state).squeeze()

        # Store trajectory data
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_values.append(value)

        # Convert action index to action list
        action_name = self.actions[action.item()]
        action_list = []
        if 'accel' in action_name:
            action_list.append('accelerate')
        if 'brake' in action_name:
            action_list.append('break')
        if 'left' in action_name:
            action_list.append('turn_left')
        if 'right' in action_name:
            action_list.append('turn_right')
        if action_name == 'accelerate':
            action_list.append('accelerate')
        elif action_name == 'brake':
            action_list.append('break')

        return action_list


    def store_reward(self, reward, done):
        reward = np.clip(reward, -100.0, 500.0)
        self.episode_rewards.append(reward)
        self.episode_dones.append(done)

    def compute_returns_and_advantages(self):
        returns = []
        advantages = []
        
        values = []
        for v in self.episode_values:
            val = v.cpu().item()
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            values.append(val)
        values.append(0.0)
        
        gae = 0
        for t in reversed(range(len(self.episode_rewards))):
            delta = (self.episode_rewards[t] + 
                    self.gamma * values[t + 1] * (1 - self.episode_dones[t]) - 
                    values[t])
            gae = delta + self.gamma * self.lam * (1 - self.episode_dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return returns, advantages

    def train(self, epochs=4):
        if self.actor is None or len(self.episode_states) < 5:
            self.clear_episode_data()
            return
        
        try:
            returns, advantages = self.compute_returns_and_advantages()
            
            states = torch.stack(self.episode_states)
            actions = torch.stack(self.episode_actions)
            old_log_probs = torch.stack(self.episode_log_probs)
            returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            
            if (torch.any(torch.isnan(states)) or torch.any(torch.isnan(actions)) or 
                torch.any(torch.isnan(old_log_probs)) or torch.any(torch.isnan(returns_tensor)) or 
                torch.any(torch.isnan(advantages_tensor))):
                self.clear_episode_data()
                return
            
            if len(advantages_tensor) > 1:
                adv_mean = advantages_tensor.mean()
                adv_std = advantages_tensor.std()
                if adv_std > 1e-6:
                    advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)

            #save weight for debugging 
            actor_weights_before = [param.clone().detach().cpu().numpy() for param in self.actor.parameters()]
            critic_weights_before = [param.clone().detach().cpu().numpy() for param in self.critic.parameters()]


            for epoch in range(epochs):
                logits = self.actor(states)  # One forward pass
                action_dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = action_dist.log_prob(actions)
                
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor
                entropy = action_dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                values = self.critic(states).squeeze()
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                if returns_tensor.dim() == 0:
                    returns_tensor = returns_tensor.unsqueeze(0)
                
                critic_loss = 0.5 * nn.functional.mse_loss(values, returns_tensor)
                
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    continue
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

            # Save weights after training (optional)
            actor_weights_after = [param.clone().detach().cpu().numpy() for param in self.actor.parameters()]
            critic_weights_after = [param.clone().detach().cpu().numpy() for param in self.critic.parameters()]

        except Exception as e:
            # Optionally print(e) for debugging
            pass
        
        finally:
            self.clear_episode_data()

    def save_models(self, actor_path="actor.pth", critic_path="critic.pth"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_models(self, actor_path="actor.pth", critic_path="critic.pth"):
        # Initialize networks (requires state_dim to be known)
        self._initialize_networks()
        
        # Load state dictionaries from disk
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()

    def clear_episode_data(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_dones = []
        
    def _update(self, reward, old_pos, action_list):
        pass