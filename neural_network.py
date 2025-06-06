import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((output_dim,))
    
    def forward(self, x):
        return np.dot(x, self.W) + self.b
    
    def backward(self, x, grad_output):
        grad_W = np.dot(x.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        grad_x = np.dot(grad_output, self.W.T)
        return grad_x, grad_W, grad_b
    
    def update(self, grad_W, grad_b, learning_rate):
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

class MeanSquaredError:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def backward(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]



class NN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = LinearLayer(input_dim, hidden_dim)
        self.layer2 = LinearLayer(hidden_dim, output_dim)
    
    def forward(self, x):
        self.x = x
        self.z1 = self.layer1.forward(x)
        # Since we need negative values for Q values, we use tanh
        self.a1 = np.tanh(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        return self.z2
    
    def backward(self, y_pred, y_true, learning_rate):
        # Simple gradient descent for MSE loss
        grad_z2 = MeanSquaredError.backward(y_true, y_pred)
        grad_a1, grad_W2, grad_b2 = self.layer2.backward(self.a1, grad_z2)

        grad_z1 = grad_a1 * (1 - np.tanh(self.z1) ** 2)  # Derivative of tanh
        #result = self.layer1.backward(self.x, grad_z1)
        #print("Backward output:",len(result))
        _, grad_W1, grad_b1 = self.layer1.backward(self.x, grad_z1)
        
        self.layer2.update(grad_W2, grad_b2, learning_rate)
        self.layer1.update(grad_W1, grad_b1, learning_rate)
    
    def predict(self, x):
        return self.forward(x)