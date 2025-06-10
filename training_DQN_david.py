from collections import deque
from DQN_agent_david import DQNAgent
from robot import Robot
import random
import numpy as np
from environment import Environment
import pygame  
from tqdm import tqdm

def train_dqn(env, episodes=500, max_steps=1000, batch_size=32, memory_size=10000):
    state_size = 6 # Get state length
    env.reset()  # Initialize the environment
    action_size = 5  # You have 5 actions

    action_idx = {
        'turn_right': 0,
        'turn_left': 1,
        'accelerate': 2,
        'break': 3,
        'break_hard': 4
    }

    agent = DQNAgent(state_size, action_size)
    memory = deque(maxlen=memory_size)

    for episode in tqdm(range(episodes), desc="Training DQN"):
        env.reset()
        state, action_list, reward = env._update(agent)
        action_list = action_idx[action_list[0]]
        total_reward = 0

        

        for step in range(max_steps):
            # You may need to modify env._update() to allow taking a specific action
            # For now, assume it uses an internal decision-making (as implied by your design)
            next_state, next_action_list, reward = env._update(agent)
            memory.append((state, action_list, reward, next_state))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                agent.train(batch)

        print(f"Episode {episode+1}/{episodes}, Reward: {env.cum_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    pygame.init()
    train_dqn(env=Environment(map_fp="map1 copy.json", agent_start_pos=(50, 50), draw=True))