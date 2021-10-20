"""Defines the soft priority queue as the SoftQueue class."""

import math
import random
from functools import *
from math import log

def main():
    """Demonstrate the behavior of SoftQueue."""
    queue = SoftQueue()
    queue.add("Hello", 7)
    queue.add("World", 5)
    queue.add("Goodbye", 3)
    queue.add("People", 2)
    for i in range(4):
        print(queue.pop())
    print(queue)

class PopInfo:
    def __init__(self, probability, total, entropy):
        self.probability = probability
        self.total = total
        self.entropy = entropy


class SoftQueue:
    """A soft priority queue. That is, a priority queue which is sampled according to the softmax of the priority."""
    class ProportionPair:
        """A helper class that functions as a named tuple."""
        def __init__(self, object, proportion):
            self.object = object
            self.proportion = proportion # The priority after being run through the exponential.

        def __str__(self):
            return f"Pair({self.object}, {self.proportion:.2f})"

    def __init__(self, sensitivity=1):
        """
        @param sensitivity: A parameter of the softmax that specifies how sensitive the proportions are with respect to the priorities.
        """
        self.queue = []
        self.total = 0 # The sum of all proportions in the queue.
        self.sensitivity = sensitivity

    def add(self, object, priority):
        """Calculates the proportion for the priority and stores the object with this proportion."""
        # Need to improve the efficiency of this using binary search
        # Well, binary search isn't needed because insertion is O(log(n)) with python lists
        proportion = math.exp(priority * self.sensitivity)
        pair = SoftQueue.ProportionPair(object, proportion)
        index = self.indexForInsertion(proportion)
        self.queue.insert(index, pair)
        self.total += proportion

    def sample(self) -> int:
        """Returns the index of an element of the queue according to the probability."""
        value = random.random()
        for i, prob in enumerate(self.probabilities()):
            value -= prob
            info = PopInfo(prob, self.total, self.entropy())
            if value < 0:
                return i, info
        return i

    def indexForInsertion(self, proportion):
        for i, pair in enumerate(self.queue):
            if proportion > pair.proportion:
                return i
        return len(self.queue)

    def pop(self):
        i, info = self.sample()
        self.total -= self.queue[i].proportion
        return self.queue.pop(i).object, info

    def __str__(self):
        if len(self.queue) > 0:
            return "SoftQueue(" + reduce(lambda x, y: f"{x}, {y}", self.queue) + ")"
        else:
            return "SoftQueue()"

    def probabilities(self):
        return map(lambda pair: pair.proportion / self.total, self.queue)

    def entropy(self):
        """Calculates the entropy from the probabilities."""
        pLogPs = map(lambda prob: -prob * log(prob), self.probabilities())
        return reduce(lambda x, y: x+y, pLogPs)

if __name__ == '__main__':
    main()