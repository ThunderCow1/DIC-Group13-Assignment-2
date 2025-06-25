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
from environment.robot import Robot
import torch 
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
class PPOAgent:
    def __init__(self, robot: Robot, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, lam=0.95, clip_eps=0.2, entropy_coef=0.02, num_sensors = 7):
        
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
        ]

        #Network dimensions
        self.state_dim = 5 + num_sensors
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

        self._initialize_networks()

    def _initialize_networks(self):
        self.actor = MLPNetwork(self.state_dim, len(self.actions), hidden_dim=128).to(self.device)
        self.critic = MLPNetwork(self.state_dim, 1, hidden_dim=128).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, x_diff, y_dff, orientation, speed, angle_diff, distances):
        state = np.array([x_diff, y_dff, orientation, speed, angle_diff] + list(distances))
        state = torch.tensor(np.float32(state)).unsqueeze(0).to(self.device)

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

        return action_name
    
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
    
    def train_(self, epochs=4):
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
                    advantages_tensor = torch.clamp(advantages_tensor, -5.0, 5.0) #trying to clip advantages

            for epoch in range(epochs):
                logits = self.actor(states)
                #logits = torch.clamp(logits, -5.0, 5.0)
                #probs = torch.sigmoid(logits)
                #probs = torch.clamp(probs, 0.1, 0.9)
                
                logits = self.actor(states)
                action_dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = action_dist.log_prob(actions)
                
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor
                entropy = action_dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                #ratios = torch.exp(new_log_probs - old_log_probs)
                #ratios = torch.clamp(ratios, 0.5, 2.0)
                
                #surr1 = ratios * advantages_tensor
                #surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor
                
                #entropy = -torch.sum(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8), dim=1).mean()
                #actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                values = self.critic(states).squeeze()
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                if returns_tensor.dim() == 0:
                    returns_tensor = returns_tensor.unsqueeze(0)
                
                critic_loss = 0.5 * nn.functional.mse_loss(values, returns_tensor)
                
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    continue
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

            #print weights for debugging 
            actor_weights_after = [param.clone().detach().cpu().numpy() for param in self.actor.parameters()]
            critic_weights_after = [param.clone().detach().cpu().numpy() for param in self.critic.parameters()]

            # Print weight differences
            # print("Actor weight changes (L2 norm):")
            # for before, after in zip(actor_weights_before, actor_weights_after):
            #     print(np.linalg.norm(after - before))

            # print("Critic weight changes (L2 norm):")
            # for before, after in zip(critic_weights_before, critic_weights_after):
            #     print(np.linalg.norm(after - before))
        except Exception as e:
            print(e)
            pass
        
        finally:
            self.clear_episode_data()

    def clear_episode_data(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_dones = []
        
    def _update(self, state, action, reward, next_state, done):
        self.store_reward(reward, done)

    def save(self, path):
        torch.save(self.actor, f"{path}_actor_ppo.pkl")
        torch.save(self.critic, f"{path}_critic_ppo.pkl")