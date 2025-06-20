import torch
import torch.nn as nn
import random
import numpy as np
from environment import Environment
from robot import Robot
from helper_functions import check_collision, action_to_index, index_to_action
from collections import deque

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN():
    def __init__(self, epsilon):
        self.actions = ['move_up',
                        'move_down',
                        'move_left',
                        'move_right']
        self.action_size = len(self.actions)
        self.main_network = NN(input_dim=16, output_dim=4)
        self.target_network = NN(input_dim=16, output_dim=4)
        self.epsilon = epsilon
        
    def select_action(self, x,y, 
                      orientation, 
                      speed, 
                      target_x, 
                      target_y, 
                      angle_diff, 
                      distances):
        dx = (target_x - x) / 1000.0
        dy = (target_y - y) / 1000.0
        state = np.array([dx, dy, angle_diff] + list(distances))
        if np.random.rand()< self.epsilon:
            return [self.actions[random.randint(0, self.action_size - 1)]]
        state = torch.Tensor(state)
        with torch.no_grad():
            q_values = self.main_network.forward(state.reshape(1, -1))
        return [self.actions[q_values.argmax().item()]]

    def _update(self,reward, old_pos, action_list):
        pass

    

