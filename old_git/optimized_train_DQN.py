from DQN import DQN
from neural_network import NN, MeanSquaredError
from robot import Robot
from environment import Environment
from collections import deque
import random
import numpy as np
import pygame
import time
import math
import copy
from tqdm import tqdm
from helper_functions import check_collision, action_to_index

# Main function to train the DQN agent
def train_dqn_agent(map_fp, 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=None, 
                    draw=True, 
                    episodes=1000,
                    max_steps=1000,
                    batch_size=32,
                    learning_rate=0.1,
                    discount_factor=0.9,
                    epsilon=0.1):

    memory = deque(maxlen = 1000)# Replay memory for DQN agent

    def optimize_model():
        
        if len(memory)< batch_size:
            print("Not enough samples in memory to optimize the model.")
            return
        
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        # Compute Q-values for current state
        q_values = dqn_agent.main_network.predict(states)

        # Predict Q-values for next state using the target network
        next_q_values = dqn_agent.target_network.predict(next_states)

        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values_full = q_values.copy()

        for i in range(batch_size):
            target_q_values_full[i, actions[i]] = rewards[i] + discount_factor * max_next_q_values[i]


        #target_q_values = rewards + discount_factor * max_next_q_values * (1 - dones)

        # print("Q-values:", q_values)
        # print("Max next Q-values:", max_next_q_values)
        # print("Target Q-values:", target_q_values)
        loss = MeanSquaredError.forward(target_q_values_full, q_values)
        #print("Loss:", loss)
        dqn_agent.main_network.backward(q_values, target_q_values_full, learning_rate)

    pygame.init()
    env = Environment(map_fp=map_fp, agent_start_pos=(50,50), draw=draw)
        
    # Initialize the DQN agent
    dqn_agent = DQN(robot=env.robot)

    # Train the agents over multiple episodes
    for episode in range(episodes):
        if episode % 10 == 0 and episode != 0:
            print(f"Episode {episode} completed. Saving models...")
            dqn_agent.save_networks()
        else:
            env.draw = False

        if episode == 0: # Init network at episode 0 only
            dqn_agent.init_networks(hidden_dim=64, output_dim=dqn_agent.action_size)

        else: # Reset environment
            env.reset(agent_start_pos=(50,50))
        
        episode_reward = 0
        steps_done = 0
        rewards_per_episode = []

        while steps_done < max_steps:
            

            # Get the current state of the environment
            state, action_list, reward = env._update(dqn_agent)

            if steps_done != 0:
                for single_action in prev_action_list:  # action is a list 
                    action_index = action_to_index(single_action)  #"FORWARD" -> 2
                    memory.append((prev_state, action_index, prev_reward, state))

            episode_reward += reward
            prev_state = np.copy(state)
            prev_action_list = np.copy(action_list)
            prev_reward = reward

            if steps_done % 10 == 0:  # Update the target network every 10 steps
                #dqn_agent.target_network = copy.deepcopy(dqn_agent.main_network)
                pass
            steps_done += 1
            # Optimize the model
            optimize_model()


        # Save the models
        # dqn_agent.save_networks()

        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward}")

    print("Training completed.")

if __name__ == "__main__":
    pygame.init()
    train_dqn_agent(map_fp="map1.json", 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=42, 
                    draw=False, 
                    episodes=1000)