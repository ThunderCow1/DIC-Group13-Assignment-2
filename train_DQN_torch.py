from DQN_torch import DQN
from neural_network import NN, MeanSquaredError
from robot2 import Robot
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
import matplotlib.pyplot as plt
import math
from collections import Counter

def normalize_state(state):
                state[2] = state[2]/180.0  # Normalize angle difference to [-1, 1]
                for i in range(3, len(state)):
                    state[i]/= 256
                return state

def soft_update(target_net, main_net, tau=0.005):
    for target_param, param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Main function to train the DQN agent
def train_dqn_agent(map_fp, 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=None, 
                    draw=True, 
                    episodes=1000,
                    batch_size= 128,
                    learning_rate=0.0001,
                    discount_factor=0.99,
                    epsilon=1):

    memory = deque(maxlen=100000)# Replay memory for DQN agent
    epsilon_min = 0.01
    epsilon_decay = 0.995
    train_iterations_per_step = 5

    dqn_agent = DQN(epsilon)
    optimizer = torch.optim.Adam(dqn_agent.main_network.parameters(), lr = learning_rate)
    def optimize_model():
        
        if len(memory)< 2 * batch_size:
            return
        
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
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
        print("Q-values taken:", q_values_taken)
        print("Target Q-values:", target_q_values)
        loss = nn.MSELoss()(q_values_taken, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    env = Environment(map_fp=map_fp, agent_start_pos=(50,50), draw=draw)
    rewards_per_episode = []
    steps_done = 0 

    # Train the agents over multiple episodes
    for episode in range(episodes):
        env.reset()             

        episode_reward = 0
        done = False
        episode_step = 0
        targets_achieved = 0
        while episode_step <= 1000 and not done:
            # Get the current state of the environment
            x, y = env.robot.position
            robot_orientation = env.robot.orientation
            robot_speed = env.robot.speed
            target_x, target_y = env.map.current_target
            direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x,y])
            angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
            # compute the angle between the robot and the target
            angle_diff = (angle_to_target - robot_orientation + 540) % 360 - 180
            distances = env.robot.gain_sensor_output(env.obstacle_mask, get_directions=False)

            # Add distance to target, angle difference, and radar distances to the state
            old_pos = env.robot.position
            old_distance = np.linalg.norm(np.array(env.map.current_target) - np.array(old_pos))

            dx = (target_x - x) / 1000.0
            dy = (target_y - y) / 1000.0
            # Normalize the state
            state = [dx, dy, angle_diff]
            for dist in distances:
                state.append(dist)
            state = normalize_state(state)
            old_pos = env.robot.position

            # Select an action using the DQN agent
            # Take the action in the environment
            next_state, action_list, reward =env._update(dqn_agent)
            next_state = normalize_state(next_state)
            collision = env.collision
            target_reached = env.target_reached

            # End episode after reaching a terminal state
            done = collision or target_reached
            if target_reached:
                 targets_achieved += 1

            #######################################################
            #REWARD FUNCTION
            new_pos = env.robot.position

            distance_r = 0
            distance_r -= 0.01 # Small penalty for each step taken
            if not collision and not target_reached:
                distance_r += 0.01 * (np.linalg.norm(np.array(env.map.current_target) - np.array(old_pos)) 
                                     - np.linalg.norm(np.array(env.map.current_target) - np.array(new_pos))) #- orientation_r
            elif collision:
                distance_r += -2.5
            elif target_reached:
                distance_r += 20
            # Add a reward
            reward = distance_r

            # print("Old Position:", old_pos,
            #       "New Position:", new_pos,
            #       "Reward:", reward)

            ##############################################################
            # Store the transition in memory
            action_index = action_to_index(action_list[0])
            if collision:
                for _ in range(5):
                    memory.append((state, action_index, reward, next_state, done))
            else:
                memory.append((state, action_index, reward, next_state, done))

            episode_reward += reward
            # Optimize the model
            for _ in range(train_iterations_per_step):
                optimize_model()

            #soft_update(dqn_agent.target_network, dqn_agent.main_network, tau=0.005)
            
            if steps_done % 100 == 0:
                dqn_agent.target_network.load_state_dict(dqn_agent.main_network.state_dict())
            
            steps_done += 1
            episode_step += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        dqn_agent.epsilon = epsilon

        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward} - Targets Achieved: {targets_achieved} - Epsilon: {epsilon:.4f}")
    torch.save(dqn_agent.main_network.state_dict(), "dqn_model.pt")
    pygame.quit()
    print("Training completed.")

    # Plot reward per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.show()


if __name__ == "__main__":
    pygame.init()
    train_dqn_agent(map_fp="map1.json", 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=42, 
                    draw=False, 
                    episodes= 2000)