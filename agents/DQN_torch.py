import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.do1 = nn.Dropout1d(0.1)
        self.fc2 = nn.Linear(32,32)
        self.do2 = nn.Dropout1d(0.1)
        self.fc3 = nn.Linear(32,output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.do1(x)
        x = torch.relu(self.fc2(x))
        x = self.do2(x)
        return self.fc3(x)
    
class DQN():
    def __init__(self, epsilon, input_size, load = False):
        self.actions = ['turn_right',
                        'turn_left',
                        'accelerate',
                        'break',
                        'break_hard']
        self.action_size = len(self.actions)
        self.main_network = NN(input_dim= input_size, output_dim= self.action_size)
        self.target_network = NN(input_dim = input_size, output_dim = self.action_size)
        self.epsilon = epsilon
        if load:
            self.load()
        
    def select_action(self, x_diff, y_dff, orientation, speed, angle_diff, distances):
        # create state vector
        state = np.array([x_diff, y_dff, orientation, speed, angle_diff] + list(distances))

        # eGreedy
        if np.random.rand()< self.epsilon:
            return [self.actions[random.randint(0, self.action_size - 1)]]
        
        state = torch.Tensor(state)

        # predict q-values
        with torch.no_grad():
            q_values = self.main_network.forward(state.reshape(1, -1))
        
        # find move with best q-value
        r = torch.argmax(q_values[0])
        
        return [self.actions[r]]
    
    def _update(self,reward, old_pos, action_list):
        pass

    def save(self):
        torch.save(self.main_network, "main_network_dqn.pkl")
        torch.save(self.target_network, "target_network_dqn.pkl")

    def load(self):
        self.main_network = torch.load("main_network_dqn.pkl")
        self.target_network = torch.load("target_network_dqn.pkl")
        self.main_network.eval()
        self.target_network.eval()
    

