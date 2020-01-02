import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.log10(1+np.exp(x))

class Activation(object):
    def __init__(self, _type):
        self.type = _type
        if (self.type == "SIGMOID"):
            self.function  = sigmoid
        elif(self.type == "RELU"):
            self.function = relu

class Layer(object):
    def __init__(self, input_dim, output_dim, activation=None):
        if (activation is None):
            self.activation = Activation("RELU")
        else:
            self.activation = Activation(activation.upper())

        self.inputs = input_dim
        self.outputs = output_dim
        self.weights = 2 * np.random.random((input_dim,output_dim)) - 1
        self.bias = np.array(2 * np.random.random((output_dim,)) - 1)
    
    def feedForward(self, inputs):
        return self.activation.function(np.dot(inputs, self.weights) + self.bias)

class NeuralNetwork(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []
    
    def addLayer(self, inputs, outputs, activation=None):
        self.layers.append(Layer(inputs, outputs, activation))

    def average(self,layers):
        for i in range(len(self.layers)):
            self.layers[i].weights = (self.layers[i].weights + layers[i].weights)/2
            self.layers[i].bias = (self.layers[i].bias + layers[i].bias)/2

    def feedForward(self, inputs):
        res = inputs
        for l in self.layers:
            res = l.feedForward(res)
        return res
