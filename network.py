import torch
from torch import nn, autograd, optim


class UniformWeighter:
    def __call__(self, features):
        return 1

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, hiddenAmount = 2):
        nn.Module.__init__(self)
        self.inputLayer = nn.Linear(inputSize, hiddenSize)
        self.hiddenLayers = nn.ModuleList([nn.Linear(hiddenSize, hiddenSize) for i in range(hiddenAmount)])
        self.outputLayer = nn.Linear(hiddenSize, 1)

    def forward(self, features):
        x = self.inputLayer(features)
        x = torch.tanh(x)
        for layer in self.hiddenLayers:
            x = layer(x)
            x = torch.tanh(x)
        x = self.outputLayer(x)
        #x = nn.LeakyReLU()(x)
        #x = torch.tanh(x)
        #x = torch.relu(x)

        return x

class TrainableNetwork:
    def __init__(self, inputSize, hiddenSize, hiddenAmount = 2):
        self.network = NeuralNetwork(inputSize, hiddenSize, hiddenAmount)
        self.optimizer = optim.Adam(params=self.network.parameters(), lr = .01)
        self.lossFunc = nn.L1Loss()

    def __call__(self, features):
        toReturn = self.network(features)
        self.network.zero_grad()
        return toReturn

    def train(self, features, desired):
        actual = self.network(features)
        loss = self.lossFunc(actual, desired)
        loss.backward()
        self.optimizer.step()
        self.network.zero_grad()

def main():
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

