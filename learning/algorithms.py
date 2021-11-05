"""
ToDo:
Integrate policy gradient.
Implement Q-Learning
"""
import copy
import random
import numpy as np
import torch

from configLoader import Configuration
from environment.softQueue import ActionInfo, StateInfo
from learning.network import TrainableNetwork
from learning.training import ReinforcementTrainer, ReinforcementAlgorithm

class SoftMC(ReinforcementAlgorithm):
    def __init__(self, isTD = True):
        self.isTD = isTD
    def networks(self):
        return ["Q"]
    def activePolicy(self):
        return "Q"

    def createRawExperience(self, trainer: ReinforcementTrainer, actionInfo: ActionInfo, stateInfo: StateInfo):
        return actionInfo, stateInfo
    def processExperienceSeed(self):
        return 0
    def processExperience(self, trainer: ReinforcementTrainer, raw, reward):
        entropyWeight = Configuration.load().entropyWeight
        discountFactor = Configuration.load().discountFactor

        actionInfo, stateInfo = raw

        reward *= discountFactor
        reward += -1 + entropyWeight * stateInfo.entropy

        processed = actionInfo.action, reward

        if self.isTD:
            reward = trainer.networks["Q"](actionInfo.action)

        return processed, reward
    def onEndOfEpisode(self, trainer: ReinforcementTrainer):
        batchSize = min(Configuration.load().replayBatchSize, len(trainer.replayBuffer))
        featuresBatch = np.zeros((batchSize, trainer.env.problem.instantiationFeatureAmount()), dtype=np.float32)
        rewardBatch = np.zeros((batchSize, 1), dtype=np.float32)

        # random.choices samples with replacement, random.sample samples without replacement
        # random.choices is cheaper, so we're using it
        i = 0
        for features, reward in random.choices(trainer.replayBuffer, k=batchSize):
            featuresBatch[i] = features
            rewardBatch[i] = reward
            i += 1

        featuresTensor = torch.from_numpy(featuresBatch)
        rewardTensor = torch.from_numpy(rewardBatch)
        trainer.networks["Q"].train(featuresTensor, rewardTensor)

class SoftSARSA(ReinforcementAlgorithm):
    def networks(self):
        return ["Q"]
    def activePolicy(self):
        return "Q"

    def createRawExperience(self, trainer: ReinforcementTrainer, actionInfo: ActionInfo, stateInfo: StateInfo):
        return actionInfo, stateInfo
    def processExperienceSeed(self):
        nextFeatures = None
        return nextFeatures
    def processExperience(self, trainer: ReinforcementTrainer, raw, nextFeatures):
        actionInfo, stateInfo = raw

        entropyWeight = Configuration.load().entropyWeight
        reward = -1 + entropyWeight * stateInfo.entropy
        processed = actionInfo.action, reward, nextFeatures # (S)AR(S)A

        return processed, actionInfo.action
    def onEndOfEpisode(self, trainer: ReinforcementTrainer):
        discountFactor = Configuration.load().discountFactor
        batchSize = min(Configuration.load().replayBatchSize, len(trainer.replayBuffer))

        featuresBatch = np.zeros((batchSize, trainer.env.problem.instantiationFeatureAmount()), dtype=np.float32)
        nextFeaturesBatch = np.zeros((batchSize, trainer.env.problem.instantiationFeatureAmount()), dtype=np.float32)
        futureWeightingBatch = np.zeros((batchSize, 1), dtype=np.float32)
        rewardBatch = np.zeros((batchSize, 1), dtype=np.float32)

        # random.choices samples with replacement, random.sample samples without replacement
        # random.choices is cheaper, so we're using it
        i = 0
        for features, reward, nextFeatures in random.choices(trainer.replayBuffer, k=batchSize):
            featuresBatch[i] = features
            rewardBatch[i] = reward
            if nextFeatures is not None:
                futureWeightingBatch[i] = discountFactor
                nextFeaturesBatch[i] = nextFeatures
            i += 1

        nextFeaturesTensor = torch.from_numpy(nextFeaturesBatch)
        futureWeightingTensor = torch.from_numpy(futureWeightingBatch)
        rewardTensor = torch.from_numpy(rewardBatch)

        with torch.no_grad():
            nextStateValuation = trainer.networks["Q"](nextFeaturesTensor)
            discountedRewards = futureWeightingTensor * nextStateValuation
            rewardTensor = discountedRewards + rewardTensor
        featuresTensor = torch.from_numpy(featuresBatch)

        trainer.networks["Q"].train(featuresTensor, rewardTensor)

class SoftQ(ReinforcementAlgorithm):
    def __init__(self):
        self.episodeCount = 0

    def networks(self):
        return ["Q1", "Q2"]
    def activePolicy(self):
        return "Q1"

    def createRawExperience(self, trainer: ReinforcementTrainer, actionInfo: ActionInfo, stateInfo: StateInfo):
        return actionInfo, stateInfo
    def processExperienceSeed(self):
        done = True
        return done
    def processExperience(self, trainer: ReinforcementTrainer, raw, done):
        actionInfo, stateInfo = raw

        entropyWeight = Configuration.load().entropyWeight
        reward = -1 + entropyWeight * stateInfo.entropy
        if not done:
            actionSpace = stateInfo.actionSpace
        else:
            actionSpace = []
        processed = actionInfo.action, reward, actionSpace

        done = False
        return processed, done
    def onEndOfEpisode(self, trainer: ReinforcementTrainer):
        discountFactor = Configuration.load().discountFactor
        argMaxSampleSize = Configuration.load().argMaxSampleSize
        batchSize = min(Configuration.load().replayBatchSize, len(trainer.replayBuffer))

        featuresBatch = np.zeros((batchSize, trainer.env.problem.instantiationFeatureAmount()), dtype=np.float32)
        nextActionSpaceBatch = np.zeros((batchSize, argMaxSampleSize, trainer.env.problem.instantiationFeatureAmount()), dtype=np.float32)
        futureWeightingBatch = np.zeros((batchSize, argMaxSampleSize, 1), dtype=np.float32)
        rewardBatch = np.zeros((batchSize, 1), dtype=np.float32)

        # random.choices samples with replacement, random.sample samples without replacement
        # random.choices is cheaper, so we're using it
        i = 0
        for features, reward, nextActionSpace in random.choices(trainer.replayBuffer, k=batchSize):
            featuresBatch[i] = features
            rewardBatch[i] = reward
            for j, actionFeatures in enumerate(nextActionSpace):
                nextActionSpaceBatch[i,j] = actionFeatures
                futureWeightingBatch[i,j] = discountFactor
            i += 1

        features = torch.from_numpy(featuresBatch)
        nextActionSpace = torch.from_numpy(nextActionSpaceBatch)
        futureWeighting = torch.from_numpy(futureWeightingBatch)
        reward = torch.from_numpy(rewardBatch)

        with torch.no_grad():
            stateValuation = futureWeighting * trainer.networks["Q1"](nextActionSpace)
            indexOfSelectedAction = torch.argmax(stateValuation, dim=1)
            selectedAction = nextActionSpace[range(batchSize), indexOfSelectedAction.flatten()]
            futureWeighting = futureWeighting[range(batchSize), indexOfSelectedAction.flatten()]
            stateValuation = futureWeighting * trainer.networks["Q2"](selectedAction)
            reward += stateValuation

        trainer.networks["Q1"].train(features, reward)

        self.swapNetworks(trainer)

    def swapNetworks(self, trainer: ReinforcementTrainer):
        self.episodeCount += 1
        if self.episodeCount > Configuration.load().swapNetworkPeriod:
            self.episodeCount = 0
            #trainer.networks["Q1"] = copy.deepcopy(trainer.networks["Q2"])
            temp = trainer.networks["Q1"]
            trainer.networks["Q1"] = trainer.networks["Q2"]
            trainer.networks["Q2"] = temp
            trainer.setActivePolicy("Q1")

def policyGradientUpdate(policy: TrainableNetwork, critic: TrainableNetwork, features, popInfo: ActionInfo):
    evaluation = critic(features)
    assert(False, "Add sensitivity of the softmax and make sure you're doing exponential correctly.")
    factor = evaluation / popInfo.total
    logP = torch.log(torch.Tensor([popInfo.probability]))
    loss = torch.mul(logP, factor)
    policy.trainByGradient(features, loss)