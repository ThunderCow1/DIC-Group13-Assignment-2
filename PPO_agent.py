import random
import numpy as np
import math
from collections import deque
from robot import Robot

class PPOAgent:
    def __init__(self, robot: Robot, 
                 learning_rate=0.0003, 
                 gamma=0.99, 
                 clip_epsilon=0.2, 
                 lam=0.95,
                 trajectory_size=2048,
                 batch_size=64,
                 optimization_epochs=10):
        self.robot = robot
        self.actions = robot.actions
        self.action_dim = len(self.actions)
        
        # State dimensions (x, y, orientation, speed, target_x, target_y, angle_diff, 5 sensor readings)
        self.state_dim = 11  
        
        # Learning parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lam = lam
        self.trajectory_size = trajectory_size
        self.batch_size = batch_size
        self.optimization_epochs = optimization_epochs
        
        # Initialize neural network weights
        self.policy_network = self._init_network(self.state_dim, self.action_dim)
        self.value_network = self._init_network(self.state_dim, 1)
        
        # Trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Counters and tracking
        self.episode_reward = 0
        self.total_steps = 0
        
    def _init_network(self, input_dim, output_dim):
        """Initialize a simple 2-layer neural network"""
        return {
            # Hidden layer (input -> hidden)
            "W1": np.random.randn(input_dim, 64) * 0.1,  
            "b1": np.zeros((64,)),
            
            # Output layer (hidden -> output)
            "W2": np.random.randn(64, output_dim) * 0.1,
            "b2": np.zeros((output_dim,))
        }
    
    def _normalize_state(self, state):
        """Normalize state values to reasonable ranges"""
        # This is a simple example - you may need to adjust based on your environment
        return np.array(state) / 800.0  # Assuming max dimension is 800
    
    def _forward(self, network, state, softmax=False):
        """Forward pass through the network"""
        # First layer
        h = state @ network["W1"] + network["b1"]
        h = np.maximum(0, h)  # ReLU activation
        
        # Output layer
        output = h @ network["W2"] + network["b2"]
        
        # Apply softmax if needed (for policy network)
        if softmax:
            output = self._softmax(output)
        
        return output
    
    def _softmax(self, x):
        """Compute softmax values for array x"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def select_action(self, x, y, robot_orientation, robot_speed, 
                  target_x, target_y, angle_diff, distances, 
                  eval_mode=False, threshold=0.3):
        """Select action(s) from the policy and return them"""
        # Create state vector
        state = [x, y, robot_orientation, robot_speed, 
                target_x, target_y, angle_diff] + list(distances)
        
        # Normalize state
        norm_state = self._normalize_state(state)
        
        # Forward pass through policy network
        action_probs = self._forward(self.policy_network, norm_state, softmax=True)
        
        # Forward pass through value network
        value = self._forward(self.value_network, norm_state)
        
        # Different action selection strategies based on mode
        if eval_mode:
            # In evaluation mode: select actions above threshold (or at least the best one)
            selected_actions = []
            for i, prob in enumerate(action_probs):
                if prob > threshold:
                    selected_actions.append(self.robot.actions[i])
            
            # If no actions are above threshold, at least take the best action
            if not selected_actions:
                best_idx = np.argmax(action_probs)
                selected_actions.append(self.robot.actions[best_idx])
                
            # Don't store trajectory info in eval mode
            return selected_actions
        else:
            # In training mode: sample a single action as before
            action_idx = np.random.choice(len(self.robot.actions), p=action_probs)
            
            # Store trajectory information
            self.states.append(norm_state)
            self.actions.append(action_idx)
            self.values.append(value[0])
            self.log_probs.append(np.log(action_probs[action_idx]))
            
            # Return the selected action
            return [self.robot.actions[action_idx]]
    
    def store_reward(self, reward, done=False):
        """Store reward and done flag for the most recent action"""
        self.rewards.append(reward)
        self.dones.append(done)
        self.episode_reward += reward
        
        if done:
            print(f"Episode finished. Total reward: {self.episode_reward}")
            self.episode_reward = 0
    
    def train(self):
        """Train the PPO agent on collected trajectory"""
        if len(self.states) < self.batch_size:
            return
        
        # Convert lists to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones, values)
        advantages = returns - values
        
        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        for _ in range(self.optimization_epochs):
            # Generate random mini-batch indices
            indices = np.random.permutation(len(states))
            
            # Process mini-batches
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Update policy network
                self._update_policy(batch_states, batch_actions, batch_log_probs, batch_advantages)
                
                # Update value network
                self._update_value(batch_states, batch_returns)
        
        # Clear trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def _compute_returns(self, rewards, dones, values):
        """Compute discounted returns with GAE (Generalized Advantage Estimation)"""
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            # If last step, next value is 0; otherwise it's the next value in our array
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # Calculate TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Accumulate GAE
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            
            # Store return
            returns[t] = gae + values[t]
            
        return returns
    
    def _update_policy(self, states, actions, old_log_probs, advantages):
        """Update policy network using PPO clipped objective"""
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            old_log_prob = old_log_probs[i]
            advantage = advantages[i]
            
            # Get current action probabilities
            action_probs = self._forward(self.policy_network, state, softmax=True)
            
            # Calculate log probability and ratio
            log_prob = np.log(action_probs[action])
            ratio = np.exp(log_prob - old_log_prob)
            
            # Calculate surrogate losses
            surrogate1 = ratio * advantage
            surrogate2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
            
            # Calculate policy gradients (for negative loss, so we use gradient ascent)
            policy_loss = -min(surrogate1, surrogate2)
            
            # Gradient of softmax+log wrt to pre-softmax outputs
            d_output = action_probs.copy()
            d_output[action] -= 1.0
            
            # Backward pass (manual backpropagation)
            d_output *= policy_loss  # Gradient of loss wrt output
            
            # Gradient for W2 and b2
            d_W2 = np.outer(self._relu_derivative(state @ self.policy_network["W1"] + self.policy_network["b1"]), d_output)
            d_b2 = d_output
            
            # Gradient for W1 and b1
            d_h = d_output @ self.policy_network["W2"].T
            d_h *= self._relu_derivative(state @ self.policy_network["W1"] + self.policy_network["b1"])
            d_W1 = np.outer(state, d_h)
            d_b1 = d_h
            
            # Update policy network parameters
            self.policy_network["W2"] -= self.lr * d_W2
            self.policy_network["b2"] -= self.lr * d_b2
            self.policy_network["W1"] -= self.lr * d_W1
            self.policy_network["b1"] -= self.lr * d_b1
    
    def _update_value(self, states, returns):
        """Update value network using MSE loss"""
        for i in range(len(states)):
            state = states[i]
            return_val = returns[i]
            
            # Forward pass
            value = self._forward(self.value_network, state)[0]
            
            # Calculate value loss gradient
            d_output = 2 * (value - return_val)
            
            # Backward pass
            # Gradient for W2 and b2
            d_W2 = np.outer(self._relu_derivative(state @ self.value_network["W1"] + self.value_network["b1"]), d_output)
            d_b2 = d_output
            
            # Gradient for W1 and b1
            d_h = d_output * self.value_network["W2"].T
            d_h *= self._relu_derivative(state @ self.value_network["W1"] + self.value_network["b1"])
            d_W1 = np.outer(state, d_h)
            d_b1 = d_h
            
            # Update value network parameters
            self.value_network["W2"] -= self.lr * d_W2
            self.value_network["b2"] -= self.lr * d_b2
            self.value_network["W1"] -= self.lr * d_W1
            self.value_network["b1"] -= self.lr * d_b1
    
    def _relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)

    def save_model(self, filepath):
        """Save model weights to file"""
        model_data = {
            "policy": self.policy_network,
            "value": self.value_network
        }
        np.save(filepath, model_data)
    
    def load_model(self, filepath):
        """Load model weights from file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.policy_network = model_data["policy"]
        self.value_network = model_data["value"]