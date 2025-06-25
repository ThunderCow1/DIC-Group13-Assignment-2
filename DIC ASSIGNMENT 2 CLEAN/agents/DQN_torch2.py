import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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

        self.train = True

        self.memory = deque(maxlen=10000)# Replay memory for DQN agent
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        
        self.loss_f = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr = 0.001)

        self.steps_done = 0

        self.last_prediction = None

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

        self.last_prediction = [str(round(x,2)) for x in q_values.numpy()[0]]
        
        # find move with best q-value
        r = torch.argmax(q_values[0])
        
        return self.actions[r]
    
    def _update(self, state, action, reward, next_state, done):
        # Update and add to memory
        self.steps_done += 1

        # Just dont bother about this
        if type(action) == list:
            self.memory.append([state, self.actions.index(action[0]), reward, next_state, done])
        else:
            self.memory.append([state, self.actions.index(action), reward, next_state, done])

        # Optimize after warmup 
        if self.steps_done > 1000 and self.train:
             self.optimize()
             if self.steps_done % 5000 == 0:
                  print('Transfering network')
                  self.target_network.load_state_dict(self.main_network.state_dict())

    def optimize(self):
        # Optimize for one batch
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q-values for current state
        q_values = self.main_network(states)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1-dones)

        # Select only the Q-values of the taken actions
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_f(q_values_taken, target_q_values)
        self.losses.append(loss.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()


    def save(self, path):
        torch.save(self.main_network, f"{path}_main_network_dqn.pkl")
        torch.save(self.target_network, f"{path}_target_network_dqn.pkl")

    def load(self, main_fp, target_fp):
        self.main_network = torch.load(main_fp)
        self.target_network = torch.load(target_fp)
        self.main_network.eval()
        self.target_network.eval()

def normalize_state(state):
                state[0], state[1] = state[0]/1000.0, state[1]/1000.0
                state[2] = (state[2] - 180)/ 180.0
                state[3] /= 10.0
                state[4] /= (state[4] - 180)/ 180.0
                for i in range(5, len(state)):
                    state[i] /= 256

                return state


    

