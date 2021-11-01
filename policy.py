"""Contains the class definition for a policy based on a soft priority queue."""
from functools import *
from network import *
from softQueue import SoftQueue
from configLoader import *

class SoftQueuePolicy:
    """A policy for picking instantiations based on a soft priority queue."""
    def __init__(self, policy = UniformWeighter()):
        config = Configuration.load()
        self.softQueue = SoftQueue(sensitivity=config.softMaxSensitivity, offset=config.softMaxOffset)
        self.addQueue = []
        self.policy = policy
    def clear(self):
        """Empties the soft queue."""
        self.softQueue = SoftQueue()
        self.addQueue.clear()
    def add(self, action, features):
        """Adds a new instantiation to the add queue."""
        self.addQueue.append((action, features))
    def __resolveAddQueue(self):
        """Adds all instantiation from the add queue to the soft queue."""
        for action, features in self.addQueue:
            self.softQueue.add(action, self.policy(features))
        self.addQueue.clear()
    def resolve(self):
        """Removes an element from the soft queue according to the probability."""
        self.__resolveAddQueue()
        return self.softQueue.pop()
    def sampleActionSpace(self, toSample: int):
        self.__resolveAddQueue()
        toSample = max(toSample, len(self.softQueue))

        sampled = []
        while toSample > 0:
            index = self.softQueue.sample()
            if not index in sampled:
                sampled.append(index)
                toSample -= 1
            # Should probably add a special sampling procedure that can't do repeats.

        return [self.softQueue[i].object for i in sampled]




