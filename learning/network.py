"""Contains the class definition for the networks which generate the priorities for the policy and that perform the valuations."""

import numpy as np
import torch
from torch import nn, autograd, optim

from configLoader import Configuration


#print(f"cuda available? {torch.cuda.is_available()}")

# I want to make this share a
class UniformWeighter:
    """A class which prioritizes every action the same, for creating a uniformly random policy."""
    def __call__(self, features):
        """Mimics how a network behaves, but outputs only ones."""
        size = features.size()
        if len(size) > 1:
            toReturn = np.empty((size[0],1))
            toReturn.fill(1)
            toReturn = torch.from_numpy(toReturn)
            return toReturn.float()
        else:
            return torch.tensor(1)

class NeuralNetwork(nn.Module):
    """A network that is to be extended with an output method."""
    def __init__(self, inputSize, hiddenSize, hiddenAmount = 2):
        nn.Module.__init__(self)
        self.inputLayer = nn.Linear(inputSize, hiddenSize)
        self.hiddenLayers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for _ in range(hiddenAmount)])

    def forward(self, features):
        """Applies the neural network."""
        x = self.inputLayer(features)
        x = torch.tanh(x)
        for layer in self.hiddenLayers:
            x = layer(x)
            x = torch.tanh(x)
        return x

class SimpleNetwork(NeuralNetwork):
    """A network with a single output and no activation function for the output layer."""
    def __init__(self, inputSize, hiddenSize, hiddenAmount = 2):
        NeuralNetwork.__init__(self, inputSize, hiddenSize, hiddenAmount)
        self.outputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, features):
        x = NeuralNetwork.forward(self, features)
        x = self.outputLayer(x)
        return x

def DuelingNetwork(NeuralNetwork):
    """A network that implements the dueling network architecture."""
    def __init__(self, inputSize, hiddenSize, hiddenAmount=2):
        NeuralNetwork.__init__(self, inputSize, hiddenSize, hiddenAmount)
        self.stateValue = nn.Linear(hiddenSize, 1)
        self.advantage = nn.Linear(hiddenSize, 1)
        raise NotImplementedError()

    def forward(self, features):
        x = NeuralNetwork.forward(self, features)
        raise NotImplementedError()
        return x

class TrainableNetwork:
    """A wrapper for a neural network that has a simpler training interface."""
    def __init__(self, inputSize, hiddenSize, hiddenAmount = 2):
        self.network = SimpleNetwork(inputSize, hiddenSize, hiddenAmount)
        config = Configuration.load()
        adamBetas = (config.adamBeta1, config.adamBeta2)
        self.optimizer = optim.Adam(params=self.network.parameters(), lr = config.learningRate, betas=adamBetas)
        self.lossFunc = nn.L1Loss()

    def __call__(self, features):
        with torch.no_grad():
            toReturn = self.network(features)
        return toReturn

    def train(self, features, desired):
        self.network.zero_grad()
        actual = self.network(features)
        loss = self.lossFunc(actual, desired)
        loss.backward()
        self.optimizer.step()

    def trainByGradient(self, features, gradient):
        self.network.zero_grad()
        self.network(features)
        gradient.backward()
        self.optimizer.step()

def main():
    """Demonstrates the behavior of TrainableNetwork."""
    torch.manual_seed(123)
    batch_size = 3
    input_size = 10
    inputFeatures = autograd.Variable(torch.rand(batch_size, input_size))
    target = autograd.Variable(torch.rand(batch_size, 1))
    policy = TrainableNetwork(input_size, 5, 2)
    print(f"target = {target}")
    print(f"inputFeatures = {inputFeatures}")

    for i in range(1000):
        policy.train(inputFeatures, target)
        if i % 100 == 0:
            print()
            print(f"out = {policy(inputFeatures)}")
    print()
    print(f"target = {target}")

if __name__ == "__main__":
    main()

