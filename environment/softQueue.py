"""Defines the soft priority queue as the SoftQueue class."""

import math
import random
from functools import reduce
from math import log
#from sortedcontainers import SortedList

def main():
    testEntropy()
def testEntropy():
    entropy = Entropy()
    probabilities = [1, .75, .3]
    for p in probabilities:
        entropy.addProbability(p)
        print(entropy)

    print("\nshould be:")
    lists = [
        [1],
        [.25,.75],
        [.25 * .7, .75 * .7,.3]
    ]
    for subList in lists:
        slowEntropy = Entropy.ofAList(subList)
        print(slowEntropy)

    print("\nundoing:")
    for i in range(len(probabilities)):
        print(f"removing {probabilities[-i-1]}")
        entropy.removeProbability(probabilities[-i - 1])
        print(entropy)
def testSoftQueue():
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
    #need to make this more numerically stable by doing softmax(x - max(x))
    class ProportionPair:
        """A helper class that functions as a named tuple."""
        def __init__(self, value, proportion):
            self.value = value
            self.proportion = proportion # The priority after being run through the exponential.
        def __str__(self):
            return f"Pair({self.value}, {self.proportion:.2f})"
        def __lt__(self, other):
            # reversing the order, so bigger probabilities are first
            return self.proportion > other.proportion

    def __init__(self, sensitivity=1, offset = 0):
        """
        @param sensitivity: A parameter of the softmax that specifies how sensitive the proportions are with respect to the priorities.
        @param offset: A parameter of the softmax that helps with numerical stability
        """
        self.queue = []
        #self.queue = SortedList()
        self.total = 0 # The sum of all proportions in the queue.
        self.sensitivity = sensitivity
        self.offset = offset
        self.entropy = Entropy()
    def clear(self):
        self.queue.clear()
        self.total = 0
        self.entropy = Entropy()

    def add(self, value, priority):
        """Calculates the proportion for the priority and stores the object with this proportion."""
        # Could improve the efficiency of this using binary search
        # Well, binary search isn't useful by itself because insertion is O(log(n)) with python lists
        proportion = math.exp((priority + self.offset) * self.sensitivity)
        pair = SoftQueue.ProportionPair(value, proportion)
        index = self.indexForInsertion(proportion)
        self.queue.insert(index, pair)
        #self.queue.add(pair)
        self.total += proportion
        self.entropy.addProbability(self.__probabilityOfProportion(proportion))
    def indexForInsertion(self, proportion):
        for i, pair in enumerate(self.queue):
            if proportion > pair.proportion:
                return i
        return len(self.queue)

    def pop(self):
        i, info = self.sample()
        proportion = self.queue[i].proportion
        self.entropy.removeProbability(self.__probabilityOfProportion(proportion))
        self.total -= proportion
        return self.queue.pop(i).value, info
    def sample(self):
        """Returns the index of an element of the queue according to the probability."""
        value = random.random()
        i = -1
        info = None
        for i, prob in enumerate(self.probabilities()):
            value -= prob
            info = PopInfo(prob, self.total, self.entropy.value)
            if value < 0:
                return i, info
        return i, info

    def __str__(self):
        if len(self.queue) > 0:
            return "SoftQueue(" + reduce(lambda x, y: f"{x}, {y}", self.queue) + ")"
        else:
            return "SoftQueue()"
    def __len__(self):
        return len(self.queue)
    def __getitem__(self, item):
        return self.queue[item]

    def probabilities(self):
        return map(lambda pair: self.__probabilityOfProportion(pair.proportion), self.queue)
    def __probabilityOfProportion(self, proportion):
        try:
            return proportion / self.total
        except ZeroDivisionError:
            print("softmax instability")
            return 1 / len(self.queue)

class Entropy:
    """Used for keeping track of the entropy of a changing list of probabilities in constant time."""
    def __init__(self):
        self.value = 0

    def addProbability(self, probability):
        if probability != 1:
            self.__scaleProbabilities(1 - probability)
            self.value += Entropy.ofAProbability(probability)
    def removeProbability(self, probability):
        if probability == 1:
            self.value = 0
            return
        self.value -= Entropy.ofAProbability(probability)
        self.__undoScaleProbabilities(1 - probability)
    def __scaleProbabilities(self, scaleFactor):
        """
        Calculates sum -p s log(p s) from H.
        Remember H = sum -p log(p)
        Assumes sum p = 1
            Which is why undoing it requires a different method
        """
        self.value *= scaleFactor
        self.value -= scaleFactor * log(scaleFactor)
    def __undoScaleProbabilities(self, scaleFactor):
        self.value += scaleFactor * log(scaleFactor)
        self.value /= scaleFactor
    def __str__(self):
        return f"Entropy({self.value})"

    @staticmethod
    def ofAList(probabilities):
        """Calculates the entropy of a list of probabilities."""
        pLogPs = map(Entropy.ofAProbability, probabilities)
        return reduce(lambda x, y: x+y, pLogPs)
    @staticmethod
    def ofAProbability(probability):
        """Shortcut for the term of a probability in the entropy calculation."""
        return -probability * log(probability)

if __name__ == '__main__':
    main()