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
    

class MultiLayerPerceptron:
    def __init__(self, layers, bias = 1.0, eta = 0.5):
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = []
        self.values = []
        self.d = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
        
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
        self.d = np.array([np.array(x) for x in self.d],dtype=object)

    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def print_weights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print()

    def run(self, x):
        x = np.array(x,dtype=object)
        self.values[0] = x
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):  
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]
    
    def bp(self, x, y):
        x = np.array(x,dtype=object)
        y = np.array(y,dtype=object)

        outputs = self.run(x)
        
        error = (y - outputs)
        MSE = sum( error ** 2) / self.layers[-1]

        self.d[-1] = outputs * (1 - outputs) * (error)

        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]): 
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]               
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i-1]+1):
                    if k==self.layers[i-1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        return MSE