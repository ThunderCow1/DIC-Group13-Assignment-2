from agents.DQN_torch import DQN
from environment.environment import Robot
from environment.environment import Environment
from collections import deque
import random
import numpy as np
import pygame
import time
import torch
import torch.nn as nn
import tqdm
from environment.helper_functions import check_collision, action_to_index
import matplotlib.pyplot as plt
import math

def normalize_state(state):
                state[0], state[1] = state[0]/1000.0, state[1]/1000.0
                state[2] /= 360.0
                state[3] /= 10.0
                state[4] /= 1000.0
                state[5] /= 1000.0
                state[6] /= 180.0
                for i in range(7,20):
                    state[i]/= 256
                return state

def soft_update_target_network(main_net, target_net, tau):
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

# Main function to train the DQN agent
def train_dqn_agent(map_fp, 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=None, 
                    draw=True, 
                    episodes=1000,
                    batch_size= 64,
                    learning_rate=0.0001,
                    discount_factor=0.999,
                    epsilon=0.2,
                    load = False):

    memory = deque(maxlen=10000)# Replay memory for DQN agent
    epsilon_min = 0.01
    epsilon_decay = 0.996
    train_iterations_per_step = 5
    loss_f = nn.MSELoss()
    dqn_agent = DQN(epsilon, load = load)
    optimizer = torch.optim.Adam(dqn_agent.main_network.parameters(), lr = learning_rate)

    def optimize_model():
        
        if len(memory)< batch_size:
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

        loss = loss_f(q_values_taken, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    env = Environment(map_fp=map_fp, agent_start_pos=(50,50), draw=draw)
    rewards_per_episode = []
    steps_done = 0 
    # Train the agents over multiple episodes
    for episode in range(episodes):
        env.reset()   
        env.robot.orientation = random.randint(0,360)
        env.robot.speed = random.uniform(0, env.robot.top_speed)

        episode_reward = 0
        done = False
        episode_step = 0
        state = None
        while not done:
            # Select an action using the DQN agent
            # Take the action in the environment
            next_state, action_list, reward =env._update(dqn_agent)
            next_state = normalize_state(next_state)
            collision = env.collision
            target_reached = env.target_reached
            done = collision or target_reached
            # End episode after 2000 steps
            done = episode_step >= 1000 or done

            #######################################################
            '''
            #REWARD FUNCTION
            new_pos = env.robot.position
            x_t, y_t = new_pos
            x_t1, y_t1 = old_pos
            x_target, y_target = env.map.current_target
            distance_to_target = np.linalg.norm(np.array(env.map.current_target) - np.array(new_pos))
            closest_obstacle_distance = min(distances)
            turn_reward = 0
            distance_r = 0
            orientation_r = 0
            # Calculate orientation reward based on the angle difference
            # between the robot's current orientation towards the target and the previous orientation
            tx = x_target - x_t1
            ty = y_target - y_t1
            dx = x_t- x_t1
            dy = y_t - y_t1
            dot = dx * tx + dy * ty
            norm_movement = np.sqrt(dx**2 + dy**2)
            norm_target = np.sqrt(tx**2 + ty**2)
            cos_theta = dot / (norm_movement * norm_target + 1e-10)
            angle = np.arccos(np.clip(cos_theta, -1, 1))
            angle_deg = np.degrees(angle)
            if x_t == x_t1 and y_t == y_t1:
                orientation_r = 0.5
            else:
                # Calculate the angle difference between the current and previous position towards the target
                # This gives us a measure of how much the robot is turning away from the target
                orientation_r = angle_deg/180.0
                 
            if not collision and not target_reached:
                if closest_obstacle_distance < 30:
                    # if sum of left side sensors is larger than right side, reward for turning left
                    if sum(distances[0:7]) > sum(distances[8:13]) and 'turn_left' in action_list:
                        turn_reward = 10
                    # if sum of right side sensors is larger than left side, reward for turning right
                    elif sum(distances[0:7]) < sum(distances[8:13]) and 'turn_right' in action_list:
                        turn_reward = 10
                    distance_r = 1- np.exp(0.4 * distance_to_target/1000) - 0.5/(closest_obstacle_distance)
                else:
                    distance_r = 1 - np.exp(0.4 * distance_to_target/1000) - orientation_r
            elif collision:
                distance_r = -20
            elif target_reached:
                distance_r = 100
            reward = distance_r + turn_reward'''
            ##############################################################
            # Store the transition in memory
            if state != None:
                for single_action in action_list: 
                    action_index = action_to_index(single_action)
                    memory.append((state, action_index, reward, next_state, done))
            episode_reward += reward
          
            # Optimize the model
            for _ in range(train_iterations_per_step):
                optimize_model()

            tau = 0.01
            soft_update_target_network(dqn_agent.main_network, dqn_agent.target_network, tau)
            
            state = next_state

            steps_done += 1
            episode_step += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        dqn_agent.epsilon = epsilon

        rewards_per_episode.append(episode_reward)
        dqn_agent.save()
        print("Saved Agent")
        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward} - Epsilon: {epsilon}")
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
    train_dqn_agent(map_fp="maps/map1.json", 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=42, 
                    draw=True, 
                    episodes= 1000,
                    load = False)