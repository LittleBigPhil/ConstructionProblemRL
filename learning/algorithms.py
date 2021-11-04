"""
ToDo:
Integrate policy gradient.
Implement Q-Learning
"""

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

def policyGradientUpdate(policy: TrainableNetwork, critic: TrainableNetwork, features, popInfo: ActionInfo):
    evaluation = critic(features)
    assert(False, "Add sensitivity of the softmax and make sure you're doing exponential correctly.")
    factor = evaluation / popInfo.total
    logP = torch.log(torch.Tensor([popInfo.probability]))
    loss = torch.mul(logP, factor)
    policy.trainByGradient(features, loss)