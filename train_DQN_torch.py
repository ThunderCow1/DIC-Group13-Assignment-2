from DQN_torch import DQN
from neural_network import NN, MeanSquaredError
from robot import Robot
from environment import Environment
from collections import deque
import random
import numpy as np
import pygame
import time
import torch
import torch.nn as nn
import tqdm
from helper_functions import check_collision, action_to_index
import math

# Main function to train the DQN agent
def train_dqn_agent(map_fp, 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=None, 
                    draw=True, 
                    episodes=1000,
                    batch_size=32,
                    learning_rate=0.001,
                    discount_factor=0.9,
                    epsilon=1):

    memory = deque(maxlen = 1000)# Replay memory for DQN agent
    epsilon_min = 0.01
    epsilon_decay = 0.995

    dqn_agent = DQN(epsilon)
    optimizer = torch.optim.Adam(dqn_agent.main_network.parameters(), lr = learning_rate)
    def optimize_model():
        
        if len(memory)< batch_size:
            return
        
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        #print(actions)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q-values for current state
        q_values = dqn_agent.main_network(states)
        
        with torch.no_grad():
            max_next_q_values = dqn_agent.target_network(next_states).max(1)[0]
            target_q_values = rewards + discount_factor * max_next_q_values * (1-dones)

        # Select only the Q-values of the taken actions
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(q_values_taken, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    env = Environment(map_fp=map_fp, agent_start_pos=(50,50), draw=draw)

    # Train the agents over multiple episodes
    for episode in range(episodes):
        pygame.init()
        env.reset(agent_start_pos=(50,50))             

        episode_reward = 0
        done = False
        steps_done = 0
        rewards_per_episode = []

        while not done:
            # Get the current state of the environment
            x, y = env.robot.position
            robot_orientation = env.robot.orientation
            robot_speed = env.robot.speed
            target_x, target_y = env.map.current_target
            direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x,y])
            angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
            angle_diff = (angle_to_target - robot_orientation + 540) % 360 - 180
            distances = env.robot.gain_sensor_output(env.obstacle_mask, get_directions=False)
            
            state = [x, y, robot_orientation, robot_speed, target_x, target_y, angle_diff]
            for dist in distances:
                state.append(dist)

            # Select an action using the DQN agent
            # Take the action in the environment
            next_state, action_list, reward =env._update(dqn_agent)

            collision = env.collision
            target_reached = env.target_reached
            done = target_reached or collision

            # Store the transition in memory
            for single_action in action_list: 
                action_index = action_to_index(single_action)
                memory.append((state, action_index, reward, next_state, done))

            episode_reward += reward

            # Optimize the model
            optimize_model()

            if steps_done % 100 == 0:  # Update the target network every 10 steps
                dqn_agent.target_network = dqn_agent.main_network
            
            steps_done += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward}")

    print("Training completed.")

if __name__ == "__main__":
    pygame.init()
    train_dqn_agent(map_fp="map1.json", 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=42, 
                    draw=True, 
                    episodes=1000)