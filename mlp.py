import numpy as np

class Perceptron:
    def __init__(self, inputs, bias=1.0):
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias


    def run(self, x):
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)
    
    def set_weights(self, w_init):
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))