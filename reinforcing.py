import torch
import numpy as np
from constructionProblem import VectorAddition
from network import *
from policy import SoftQueuePolicy
from reteEnvironment import ReteEnvironment
from softQueue import PopInfo
import matplotlib.pyplot as plt
from configLoader import *
from statistics import *

class ReinforcementTrainer:
    pass

def rollout(env: ReteEnvironment, policy: TrainableNetwork, critic: TrainableNetwork, maxStepsPerEpisode: int):
    """Generates raw experiences by stepping according to the policy."""
    experienceStack = []
    for i in range(maxStepsPerEpisode):
        done, inferred, features, popInfo = env.step()
        experienceStack.append((features, popInfo.entropy))
        if done:
            break
        #policyGradientUpdate(policy, critic, features, popInfo)
    return experienceStack

def policyGradientUpdate(policy: TrainableNetwork, critic: TrainableNetwork, features, popInfo: PopInfo):
    evaluation = critic(features) - 1 # not sure if the -1 is appropriate
    factor = evaluation / popInfo.total
    logP = torch.log(torch.Tensor([popInfo.probability]))
    loss = torch.mul(logP, factor)
    policy.trainByGradient(features, loss)

def makeExperiencesMC(experienceStack):
    """Propagates the reward back to transform raw experiences into informed experiences."""
    """ToDo:
    Add support for non-monte-carlo return."""
    entropyWeight = Configuration.load().entropyWeight
    discountFactor = Configuration.load().discountFactor
    replayBuffer = []
    reward = 0
    for i in range(len(experienceStack)):
        stackIndex = len(experienceStack) - i - 1
        features, entropy = experienceStack[stackIndex]
        reward *= discountFactor
        reward += -1 + entropyWeight * entropy
        replayBuffer.append((features, reward))
    return replayBuffer

def dreamMC(policy, replayBuffer):
    """Trains from the already generated experiences."""
    """ToDo:
    Create minibatches from the replay buffer and train on those.
    Allow for policy gradient instead of value learning."""
    for features, reward in replayBuffer:
        reward = torch.tensor([reward])
        policy.train(features, reward)

def main():
    problem = VectorAddition()
    inputSize = problem.featureAmount()

    #innerPolicy = UniformWeighter()

    hiddenLayers = Configuration.load().hiddenLayers
    hiddenLayerSizeFactor = Configuration.load().hiddenLayerSizeFactor

    innerPolicy = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
    outerPolicy = SoftQueuePolicy(policy=innerPolicy)
    critic = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
    env = ReteEnvironment(problem=VectorAddition(), studentPolicy=outerPolicy)

    maxStepsPerEpisode = Configuration.load().maxStepsPerEpisode

    yVals = []
    xVals = []

    quality = BiasCorrectedMomentum(Configuration.load().qualityMomentum)
    bigStep = quality.timeScale()

    i = 0
    while True:
        i += 1
        experienceStack = rollout(env, innerPolicy, critic, maxStepsPerEpisode)

        quality.add(maxStepsPerEpisode - len(experienceStack))
        if i > bigStep:
            xVals.append(i)
            yVals.append(quality.get())

        replayBuffer = makeExperiencesMC(experienceStack)
        dreamMC(innerPolicy, replayBuffer)
        #dreamMC(critic, replayBuffer)
        env.reset()
        if i > bigStep * 2 and i % (bigStep * 4) == 0:
            plt.plot(xVals, yVals)
            plt.show()


if __name__ == '__main__':
    main()