import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights)+ self.biases
        



X,y = spiral_data(100,3)

dense1 = Layer_Dense(2,3)

dense1.forward(X)   
     
print(dense1.output[:5])
