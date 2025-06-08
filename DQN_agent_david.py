import numpy as np
import random

class DQNAgent:
    
    def __init__(self,
                 hidden_size=64, 
                 learning_rate=0.001, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_min=0.1, epsilon_decay=0.995):
        self.input_size = 12
        self.action_size = 5
        self.actions = ['turn_right',
                        'turn_left',
                        'accelerate',
                        'break',
                        'break_hard']

        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Create random weights for Linear layers
        self.W1 = np.random.randn(self.input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, self.action_size) * 0.01
        self.b2 = np.zeros((1, self.action_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def predict(self, state):
        # Calculate layer 1
        z1 = np.dot(state, self.W1) + self.b1

        # Activation
        a1 = self.relu(z1)

        # Calculate layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        return z2  # Predicted Q-values(??)

    def select_action(self, x, y, orientation, speed, target_x, target_y, angle_diff, distances):
        state = np.array([x, y, orientation, speed, target_x, target_y, angle_diff] + list(distances))
        state = state.reshape(1, -1)  # Reshape to match input size
        if np.random.rand() < self.epsilon:
            return [self.actions[random.randint(0, self.action_size - 1)]]
        
        q_values = self.predict(state.reshape(1, -1))
        r = np.argmax(q_values[0])
        return [self.actions[r]]

    def train(self, batch):
        states, actions, rewards, next_states = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Forward pass (store a1 and z1 for backward pass)
        z1 = np.dot(states, self.W1) + self.b1
        a1 = self.relu(z1)
        q_values = np.dot(a1, self.W2) + self.b2

        # Target Q-values
        next_q = self.predict(next_states)
        target_q_values = q_values.copy()

        for i in range(len(batch)):
            best_next_q = np.max(next_q[i])
            target = rewards[i] + self.gamma * best_next_q
            target_q_values[i, actions[i]] = target

        np.clip(target_q_values, -100, 100, out=target_q_values)
        # Backward pass
        loss_grad = 2 * (q_values - target_q_values) / len(batch)

        # Calculate gradients
        dW2 = np.dot(a1.T, loss_grad)
        db2 = np.sum(loss_grad, axis=0, keepdims=True)

        da1 = np.dot(loss_grad, self.W2.T)
        dz1 = da1 * self.relu_deriv(z1)

        dW1 = np.dot(states.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Clip gradients to prevent exploding gradients
        dW1 = np.clip(dW1, -1, 1)
        dW2 = np.clip(dW2, -1, 1)
        db1 = np.clip(db1, -1, 1)
        db2 = np.clip(db2, -1, 1)
        

        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _update(self, reward, old_pos, action_list):
        pass
