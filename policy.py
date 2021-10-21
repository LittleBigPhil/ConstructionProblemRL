"""Contains the class definition for a policy based on a soft priority queue."""
from functools import *
from network import *
from softQueue import SoftQueue

class SoftQueuePolicy:
    """A policy for picking instantiations based on a soft priority queue."""
    def __init__(self, policy = UniformWeighter()):
        self.queue = SoftQueue()
        self.policy = policy
    def clear(self):
        """Empties the queue."""
        self.queue = SoftQueue()
    def add(self, action, features):
        """Adds a new instantiation to the queue."""
        self.queue.add(action, self.policy(features))
    def resolve(self):
        """Removes an element from the queue according to the probability."""
        return self.queue.pop()
    def sampleActionSpace(self, toSample: int):
        toSample = max(toSample, len(self.queue))

        sampled = []
        while toSample > 0:
            index = self.queue.sample()
            if not index in sampled:
                sampled.append(index)
                toSample -= 1
            # Should probably add a special sampling procedure that can't do repeats.

        return [self.queue[i].object for i in sampled]




