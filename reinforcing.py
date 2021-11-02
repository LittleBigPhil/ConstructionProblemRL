import torch
import numpy as np

from network import *
from policy import SoftQueuePolicy
from softQueue import PopInfo

from reteEnvironment import ReteEnvironment
from constructionProblem import VectorAddition

from configLoader import *

from collections import deque
from reinforcementAlgorithm import *

import matplotlib.pyplot as plt
from statistics import *



class ReinforcementTrainer:
    def __init__(self):
        problem = VectorAddition()
        inputSize = problem.instantiationFeatureAmount()

        hiddenLayers = Configuration.load().hiddenLayers
        hiddenLayerSizeFactor = Configuration.load().hiddenLayerSizeFactor

        # innerPolicy = UniformWeighter()
        self.innerPolicy = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
        self.outerPolicy = SoftQueuePolicy(policy=self.innerPolicy)
        self.critic = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
        self.env = ReteEnvironment(problem=problem, studentPolicy=self.outerPolicy)
        self.algorithm = SoftMC()

        self.replayBuffer = deque(maxlen = Configuration.load().replayBufferSize) # Circular buffer

    def rollout(self, maxStepsPerEpisode: int) -> int:
        experienceStack = self.__makeRawExperiences(maxStepsPerEpisode)
        self.__processExperiences(experienceStack)
        self.algorithm.onEndOfEpisode(self)
        return len(experienceStack)

    def __makeRawExperiences(self, maxStepsPerEpisode: int):
        """Generates raw experiences by stepping according to the policy."""
        experienceStack = []
        for _ in range(maxStepsPerEpisode):
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