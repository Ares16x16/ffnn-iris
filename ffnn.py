import numpy as np

class FFNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_hat = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        return y_hat
    
    def backprop(self, X, y, y_hat, learning_rate):
        m = X.shape[0]
        delta3 = y_hat - y
        delta2 = np.dot(delta3, self.W2.T) * (1 - np.power(self.a1, 2))
        dW2 = np.dot(self.a1.T, delta3) / m
        db2 = np.sum(delta3, axis=0, keepdims=True) / m
        dW1 = np.dot(X.T, delta2) / m
        db1 = np.sum(delta2, axis=0) / m
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            y_hat = self.forward(X)
            self.backprop(X, y, y_hat, learning_rate)
            loss = self.loss(y, y_hat)
            if i % 100 == 0:
                print(f"Epoch {i}, loss={loss:.4f}")
                
    def loss(self, y, y_hat):
        correct_logprobs = -np.log(y_hat[range(len(y_hat)), np.argmax(y, axis=1)])
        loss = np.sum(correct_logprobs) / len(y)
        return loss