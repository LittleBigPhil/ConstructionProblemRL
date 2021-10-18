from functools import *
from math import log
from network import *
from softQueue import SoftQueue

class SoftQueuePolicy:
    def __init__(self, policy = UniformWeighter()):
        self.queue = SoftQueue()
        self.policy = policy
    def clear(self):
        self.queue = SoftQueue()
    def add(self, action, features):
        self.queue.add(action, self.policy(features))
    def resolve(self):
        return self.queue.pop()

    def entropy(self):
        pLogPs = map(lambda prob: -prob * log(prob), self.queue.probabilities())
        return reduce(lambda x, y: x+y, pLogPs)


