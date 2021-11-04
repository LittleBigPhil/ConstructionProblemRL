from collections import deque

from configLoader import Configuration
from environment.constructionProblem import VectorAddition
from environment.reteEnvironment import ReteEnvironment
from learning.network import TrainableNetwork
from learning.policy import SoftQueuePolicy
from environment.softQueue import ActionInfo, StateInfo


class ReinforcementTrainer:
    def __init__(self, algorithm: "ReinforcementAlgorithm"):
        problem = VectorAddition()
        inputSize = problem.instantiationFeatureAmount()

        hiddenLayers = Configuration.load().hiddenLayers
        hiddenLayerSizeFactor = Configuration.load().hiddenLayerSizeFactor

        # innerPolicy = UniformWeighter()
        def makeNetwork():
            return TrainableNetwork(inputSize, inputSize * hiddenLayerSizeFactor, hiddenLayers)

        self.__algorithm = algorithm
        self.networks = {name: makeNetwork() for name in self.__algorithm.networks()}
        self.__outerPolicy = SoftQueuePolicy(policy=self.networks[self.__algorithm.activePolicy()])
        self.env = ReteEnvironment(problem=problem, studentPolicy=self.__outerPolicy)

        self.replayBuffer = deque(maxlen = Configuration.load().replayBufferSize) # Circular buffer

    def rollout(self, maxStepsPerEpisode: int) -> int:
        experienceStack = self.__makeRawExperiences(maxStepsPerEpisode)
        self.__processExperiences(experienceStack)
        self.__algorithm.onEndOfEpisode(self)
        return len(experienceStack)

    def __makeRawExperiences(self, maxStepsPerEpisode: int):
        """Generates raw experiences by stepping according to the policy."""
        experienceStack = []
        for _ in range(maxStepsPerEpisode):
            done, actionInfo, stateInfo = self.env.step()
            experienceStack.append(self.__algorithm.createRawExperience(self, actionInfo, stateInfo))
            if done:
                break
        self.env.reset()
        return experienceStack

    def __processExperiences(self, experienceStack):
        """Propagates the reward back to transform raw experiences into informed experiences."""
        seed = self.__algorithm.processExperienceSeed()
        for i in range(len(experienceStack)):
            stackIndex = len(experienceStack) - i - 1

            raw = experienceStack[stackIndex]
            processed, seed = self.__algorithm.processExperience(self, raw, seed)
            self.replayBuffer.append(processed)

    def setActivePolicy(self, name: str):
        self.__outerPolicy.policy = self.networks[name]

class ReinforcementAlgorithm:
    def networks(self):
        raise NotImplementedError()
    def activePolicy(self):
        raise NotImplementedError()
    def createRawExperience(self, trainer: ReinforcementTrainer, actionInfo: ActionInfo, stateInfo: StateInfo):
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
