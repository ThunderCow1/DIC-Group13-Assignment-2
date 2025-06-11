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
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN():
    def __init__(self, epsilon):
        self.actions = ['turn_right',
                        'turn_left',
                        'accelerate',
                        'break',
                        'break_hard']
        self.action_size = len(self.actions)
        self.main_network = NN(input_dim=12, output_dim=5)
        self.target_network = NN(input_dim = 12, output_dim = 5)
        self.epsilon = epsilon
        
    def select_action(self, x,y, 
                      orientation, 
                      speed, 
                      target_x, 
                      target_y, 
                      angle_diff, 
                      distances):
        state = np.array([x, y, orientation, speed, target_x, target_y, angle_diff] + list(distances))
        if np.random.rand()< self.epsilon:
            return [self.actions[random.randint(0, self.action_size - 1)]]
        state = torch.Tensor(state)
        q_values = self.main_network.forward(state.reshape(1, -1))
        r = torch.argmax(q_values[0])
        return [self.actions[r]]
    
    def _update(self,reward, old_pos, action_list):
        pass

    

