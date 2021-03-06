"""Contains the class definition for a policy based on a soft priority queue."""
import numpy as np
import torch

from configLoader import Configuration
from environment.softQueue import SoftQueue, StateInfo, ActionInfo
from learning.network import UniformWeighter

class SoftQueuePolicy:
    """A policy for picking instantiations based on a soft priority queue."""
    def __init__(self, policy = UniformWeighter()):
        config = Configuration.load()
        self.softQueue = SoftQueue(sensitivity=config.softMaxSensitivity, offset=config.softMaxOffset)
        self.addQueue = []
        self.policy = policy
    def clear(self):
        """Empties the queues."""
        self.softQueue.clear()
        self.addQueue.clear()

    def add(self, action, features):
        """Adds a new instantiation to the add queue."""
        self.addQueue.append((action, features))
    def __resolveAddQueue(self):
        """Adds all instantiation from the add queue to the soft queue."""
        if len(self.addQueue) > 0:
            _, features = self.addQueue[0]
            featuresBatch = np.zeros((len(self.addQueue), len(features)), dtype=np.float32)
            i = 0
            for _, features in self.addQueue:
                featuresBatch[i] = features
                i+=1

            featuresTensor = torch.from_numpy(featuresBatch)

            prioritiesTensor = self.policy(featuresTensor)
            for i, (action, _) in enumerate(self.addQueue):
                self.softQueue.add(action, prioritiesTensor[i,0].item())


        self.addQueue.clear()
    def resolve(self) -> ActionInfo:
        """Removes an element from the soft queue according to the probability."""
        self.__resolveAddQueue()
        return self.softQueue.pop()
    def state(self) -> StateInfo:
        sampleSize = Configuration.load().argMaxSampleSize
        return StateInfo(self.softQueue.entropy.value, self.__sampleActionSpace(sampleSize))
    def __sampleActionSpace(self, toSample: int):
        self.__resolveAddQueue()
        toSample = min(toSample, len(self.softQueue))

        # Samples without repeats.
        sampled = []
        while toSample > 0:
            index = self.softQueue.sample().action
            if not index in sampled:
                sampled.append(index)
                toSample -= 1

        # Samples with repeats.
        #sampled = [self.softQueue.sample().action for _ in range(toSample)]

        return [self.softQueue[i].value for i in sampled]





