"""Contains the class definition for a policy based on a soft priority queue."""
from functools import *
from math import log
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

    def entropy(self):
        """Calculates the entropy of the policy from the probabilities of the queue."""
        pLogPs = map(lambda prob: -prob * log(prob), self.queue.probabilities())
        return reduce(lambda x, y: x+y, pLogPs)


