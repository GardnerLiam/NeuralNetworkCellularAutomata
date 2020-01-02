from network import NeuralNetwork
import numpy as np

class Cell(object):
    def __init__(self, x, y, s, maxX, maxY):
        self.NN = NeuralNetwork(20)
        self.x = x
        self.y = y
        self.s = np.random.random()

        self.maxX = maxX
        self.maxY = maxY

        self.NN.addLayer(20,9,"RELU")
        self.NN.addLayer(9,4,"SIGMOID")
        

        self.life = 6
        self.isDead = False

    def move(self, grid):
        movement_stats = self.NN.feedForward(np.array(grid).flatten())
        move = np.argmax(movement_stats)
        if (move == 0):
            if (self.y == 0):
                self.y = self.maxY
            else:
                self.y -=1
        if (move == 1):
            if (self.x == self.maxX):
                self.x = 0
            else:
                self.x +=1
        if (move == 2):
            if (self.y == self.maxY):
                self.y = 0
            else:
                self.y +=1
        if (move == 3):
            if (self.x == 0):
                self.x = self.maxX
            else:
                self.x -=1
        self.life -= 1
    
    def birth(self, x, y):
        c_strength = self.s
        if (np.random.random() < 0.1):
            c_strength += (2*np.random.random()-1)/10

        c_nn = self.NN
        layers = c_nn.layers[:]

        for i in layers:
            if (np.random.random() < 0.1):
                i.weights = 2*np.random.random(i.weights.shape)-1
                i.bias = 2*np.random.random()-1
            else:
                i.weights = np.zeros(i.weights.shape)
                try:
                    i.bias = np.zeros(i.bias.shape)
                except:
                    i.bias = 0
        c_nn.average(layers)

        child = Cell(x,y,c_strength,self.maxX,self.maxY)
        child.NN = c_nn
        return child
                


