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
from helper_functions import check_collision, action_to_index

# Main function to train the DQN agent
def train_dqn_agent(map_fp, 
                    no_gui=False, 
                    target_fps=30, 
                    random_seed=None, 
                    draw=False, 
                    episodes=1000,
                    batch_size=32,
                    learning_rate=0.01,
                    discount_factor=0.9,
                    epsilon=0.1):

    memory = deque(maxlen = 1000)# Replay memory for DQN agent

    def optimize_model():
        
        if len(memory)< batch_size:
            return
        
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        #print(actions)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Compute Q-values for current state
        q_values = dqn_agent.main_network.predict(states)
        next_q_values = dqn_agent.target_network.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values_full = q_values.copy()
        for i in range(batch_size):
            target_q_values_full[i, actions[i]] = rewards[i] + discount_factor * max_next_q_values[i] * (1 - dones[i])


        #target_q_values = rewards + discount_factor * max_next_q_values * (1 - dones)

        # print("Q-values:", q_values)
        # print("Max next Q-values:", max_next_q_values)
        # print("Target Q-values:", target_q_values)
        loss = MeanSquaredError.forward(target_q_values_full, q_values)
        #print("Loss:", loss)
        dqn_agent.main_network.backward(q_values, target_q_values_full, learning_rate)

    # Train the agents over multiple episodes
    rewards_per_episode = []
    steps_done = 0
    for episode in range(episodes):
        pygame.init()
        env = Environment(map_fp=map_fp, no_gui=no_gui, agent_start_pos=(50,50), target_fps=target_fps, random_seed=random_seed, draw=draw)
        
        # Initialize the DQN agent
        dqn_agent = DQN(robot=env.robot)
        episode_reward = 0
        done = False

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
            action = dqn_agent.select_action(x, y, robot_orientation, robot_speed, target_x, target_y, angle_diff, distances, epsilon)

            # Take the action in the environment
            env.robot.take_action([action])
            env._update(dqn_agent)

            # Get the next state and reward
            x, y = env.robot.position
            robot_orientation = env.robot.orientation
            robot_speed = env.robot.speed
            target_x, target_y = env.map.current_target
            direction_vec = pygame.Vector2([target_x, target_y]) - pygame.Vector2([x,y])
            angle_to_target = math.degrees(math.atan2(direction_vec.y, direction_vec.x)) % 360
            angle_diff = (angle_to_target - robot_orientation + 540) % 360 - 180
            distances = env.robot.gain_sensor_output(env.obstacle_mask, get_directions=False)

            next_state = [x, y, robot_orientation, robot_speed, target_x, target_y, angle_diff]
            for dist in distances:
                next_state.append(dist)

            print("Position:", env.robot.position)
            collision = check_collision(env.robot.position, env.robot.size, env.obstacle_mask)
            target_reached = env.check_target(env.robot.position, env.robot.size)
            reward = env.reward_function(collision=collision, target_reached=target_reached)
            print("Collision:", collision, "Target Reached:", target_reached, "Reward:", reward)
            done = target_reached or collision


            # Store the transition in memory
            for single_action in action:  # action is a list 
                action_index = action_to_index(single_action)  #"FORWARD" -> 2
                memory.append((state, action_index, reward, next_state, done))

            episode_reward += reward

            if steps_done % 10 == 0:  # Update the target network every 10 steps
                dqn_agent.target_network = dqn_agent.main_network

            # Optimize the model
            optimize_model()
            steps_done += 1

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