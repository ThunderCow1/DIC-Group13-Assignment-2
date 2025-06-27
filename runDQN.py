import numpy as np
from argparse import ArgumentParser
import json
import pygame
import time
import random
import torch

# Agents
#from agents.DQN_torch import DQN as DQNAgent
from agents.DQN_torch2 import DQN as DQNAgent

# from agents.PPO_agent import PPOAgent
# from agents.human_agent import HumanAgent
# from agents.intuitive_agent import IntuitiveAgent


# Environment
from environment.environment import Environment


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    
    

    m = 'maps\map2.json'
    env = Environment(m, random_seed=42, draw=True)
    num_sensors = 7
    agent = DQNAgent(0.25, 5 + num_sensors)
    agent.load('networks_great_success (somewhat)\epoch19_main_network_dqn.pkl', 'networks_great_success (somewhat)\epoch19_target_network_dqn.pkl')

    active = True
    agent.train = False

    i = 0
    while active:
        env._update(agent=agent)
        i += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                active = False
        
        if i % 1000 == 0:
            env.reset()


    

    

    
    
