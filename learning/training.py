from collections import deque

from configLoader import Configuration
from environment.constructionProblem import VectorAddition
from environment.reteEnvironment import ReteEnvironment
from learning.network import TrainableNetwork
from learning.policy import SoftQueuePolicy


class ReinforcementTrainer:
    def __init__(self, algorithm):
        problem = VectorAddition()
        inputSize = problem.instantiationFeatureAmount()

        hiddenLayers = Configuration.load().hiddenLayers
        hiddenLayerSizeFactor = Configuration.load().hiddenLayerSizeFactor

        # innerPolicy = UniformWeighter()
        self.innerPolicy = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
        self.outerPolicy = SoftQueuePolicy(policy=self.innerPolicy)
        self.critic = TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)
        self.env = ReteEnvironment(problem=problem, studentPolicy=self.outerPolicy)
        self.algorithm = algorithm

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
