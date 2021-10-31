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
from collections import deque
from random import randrange

"""
ToDo:
Make replay buffer minibatches compatible with batch training.
Integrate policy gradient.
"""

class ReinforcementTrainer:
    def __init__(self):
        problem = VectorAddition()
        inputSize = problem.featureAmount()

        hiddenLayers = Configuration.load().hiddenLayers
        hiddenLayerSizeFactor = Configuration.load().hiddenLayerSizeFactor

        # innerPolicy = UniformWeighter()
        self.innerPolicy = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
        self.outerPolicy = SoftQueuePolicy(policy=self.innerPolicy)
        self.critic = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
        self.env = ReteEnvironment(problem=problem, studentPolicy=self.outerPolicy)
        self.algorithm = SoftQ()

        self.replayBuffer = deque(maxlen = Configuration.load().replayBufferSize) # Circular buffer

    def rollout(self, maxStepsPerEpisode: int) -> int:
        experienceStack = self.__makeRawExperiences(maxStepsPerEpisode)
        self.__processExperiences(experienceStack)
        self.algorithm.onEndOfEpisode(self)
        return len(experienceStack)

    def __makeRawExperiences(self, maxStepsPerEpisode: int):
        """Generates raw experiences by stepping according to the policy."""
        experienceStack = []
        for i in range(maxStepsPerEpisode):
            done, inferred, features, popInfo = self.env.step()
            experienceStack.append(self.algorithm.createRawExperience(self, features, popInfo))
            if done:
                break
        self.env.reset()
        return experienceStack

    def __processExperiences(self, experienceStack):
        """Propagates the reward back to transform raw experiences into informed experiences."""
        seed = self.algorithm.processExperienceSeed()
        for i in range(len(experienceStack)):
            stackIndex = len(experienceStack) - i - 1

            raw = experienceStack[stackIndex]
            processed, seed = self.algorithm.processExperience(self, raw, seed)
            self.replayBuffer.append(processed)


class ReinforcementAlgorithm:
    def createRawExperience(self, trainer: ReinforcementTrainer, features, popInfo):
        raise NotImplementedError()
    def processExperienceSeed(self):
        """Returns the seed for use in processing raw experiences."""
        raise NotImplementedError()
    def processExperience(self, trainer: ReinforcementTrainer, raw, seed):
        """
        raw, seed -> processed, seed
        Steps backward one step in processing the raw experiences.
        """
        raise NotImplementedError()
    def onEndOfEpisode(self, trainer: ReinforcementTrainer):
        raise NotImplementedError()

class SoftQ(ReinforcementAlgorithm):
    def __init__(self, isTD = True):
        self.isTD = isTD
    def createRawExperience(self, trainer: ReinforcementTrainer, features, popInfo):
        return features, popInfo.entropy
    def processExperienceSeed(self):
        return 0
    def processExperience(self, trainer: ReinforcementTrainer, raw, reward):
        entropyWeight = Configuration.load().entropyWeight
        discountFactor = Configuration.load().discountFactor

        features, entropy = raw

        reward *= discountFactor
        reward += -1 + entropyWeight * entropy

        processed = features, reward


        if self.isTD:
            reward = trainer.innerPolicy(features)

        return processed, reward
    def onEndOfEpisode(self, trainer: ReinforcementTrainer):
        featuresBatch = []
        rewardBatch = []
        for _ in range(Configuration.load().replayBatchSize):
            index = randrange(0, len(trainer.replayBuffer))
            features, reward = trainer.replayBuffer[index]

            reward = torch.tensor([reward])
            featuresBatch.append(torch.unsqueeze(features, 0))
            rewardBatch.append(torch.unsqueeze(reward, 0))
            #trainer.innerPolicy.train(features, reward)
        featuresTensor = torch.cat(featuresBatch)
        rewardTensor = torch.cat(rewardBatch)
        #print(featuresTensor)
        #print(rewardTensor)
        trainer.innerPolicy.train(featuresTensor, rewardTensor)

def policyGradientUpdate(policy: TrainableNetwork, critic: TrainableNetwork, features, popInfo: PopInfo):
    evaluation = critic(features)
    assert(False, "Add sensitivity of the softmax and make sure you're doing exponential correctly.")
    factor = evaluation / popInfo.total
    logP = torch.log(torch.Tensor([popInfo.probability]))
    loss = torch.mul(logP, factor)
    policy.trainByGradient(features, loss)

def main(maxI: int = -1):
    rlTrainer = ReinforcementTrainer()
    maxStepsPerEpisode = Configuration.load().maxStepsPerEpisode

    yVals = []
    xVals = []

    quality = BiasCorrectedMomentum(Configuration.load().qualityMomentum)
    bigStep = quality.timeScale()

    i = 0
    while maxI < 0 or i < maxI:
        i += 1
        length = rlTrainer.rollout(maxStepsPerEpisode)

        quality.add(maxStepsPerEpisode - length)
        if i > bigStep:
            xVals.append(i)
            yVals.append(quality.get())
        if i > bigStep * 2 and i % (bigStep * 4) == 0:
            plt.plot(xVals, yVals)
            plt.show()


if __name__ == '__main__':
    main()