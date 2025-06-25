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
from neural_network import NN


class DQN():
    def __init__(self, 
                 robot: Robot):
        self.robot = robot
        self.actions = robot.actions
        self.position = robot.position
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.state_size = 7
        self.action_size = len(self.actions)
        self.collision = False
        self.target_reached = False
    
    def init_networks(self, hidden_dim, output_dim):
        self.main_network = NN(self.state_size + self.action_size, hidden_dim, output_dim)
        self.target_network = NN(self.state_size + self.action_size, hidden_dim, output_dim)

    def save_networks(self):
        self.main_network.save("main_net_DQN.npz")
        self.target_network.save("target_net_DQN.npz")

    def load_networks(self, hidden_dim, output_dim, main_network, target_network):
        self.main_network = NN(self.state_size + self.action_size, hidden_dim, output_dim)
        self.target_network = NN(self.state_size + self.action_size, hidden_dim, output_dim)
        self.main_network.load(main_network)
        self.target_network.load(target_network)

    def update_state(self, x, y, robot_orientation, robot_speed, target_x, target_y):
        self.state = (x, y, robot_orientation, robot_speed, target_x, target_y)

    def select_action(self,
                      x,
                      y,
                      robot_orientation,
                      robot_speed,
                      target_x,
                      target_y,
                      angle_diff,
                      distances,
                      epsilon = 0.1):
        # Input of the nn is the state of the robots and the distances combined. 
        input_vector = [x, y, robot_orientation, robot_speed, target_x, target_y, angle_diff] 
        for i in distances:
            input_vector.append(i)
        input_vector = np.array(input_vector)

        if random.random() < epsilon:
            num_actions = random.randint(1, len(self.actions))
            # Randomly select a number of actions from the available actions
            return random.sample(self.actions, num_actions)
        else:
            q_values = self.main_network.predict(input_vector)
            # Get indexes with the positive Q value, return actions with the positive indices
            positive_q_indices = np.where(q_values > 0)[0]
            if positive_q_indices.size > 0:
                return [self.actions[i] for i in positive_q_indices]
            else:
                num_actions = random.randint(1, len(self.actions))
                return random.sample(self.actions, num_actions)

    def _update(self,reward, old_pos, action_list):
        pass