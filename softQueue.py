import math
import random
from functools import *

def main():
    queue = SoftQueue()
    queue.add("Hello", 7)
    queue.add("World", 5)
    queue.add("Goodbye", 3)
    queue.add("People", 2)
    #for i in range(4):
    #    print(queue.pop())
    print(queue)

class SoftQueue:
    class ProportionPair:
        def __init__(self, object, proportion):
            self.object = object
            self.proportion = proportion

        def __str__(self):
            return f"Pair({self.object}, {self.proportion:.2f})"

    def __init__(self, sensitivity=1):
        self.queue = []
        self.total = 0
        self.sensitivity = sensitivity

    def add(self, object, proportion):
        # Need to improve the efficiency of this using binary search
        # Well, binary search isn't needed because insertion is O(log(n)) with python lists
        proportion = math.exp(proportion * self.sensitivity)
        pair = SoftQueue.ProportionPair(object, proportion)
        index = self.indexForInsertion(proportion)
        self.queue.insert(index, pair)
        self.total += proportion

    def sample(self):
        value = random.random()
        for i, prob in enumerate(self.probabilities()):
            value -= prob
            if value < 0:
                return i
        return i

    def indexForInsertion(self, proportion):
        for i, pair in enumerate(self.queue):
            if proportion > pair.proportion:
                return i
        return len(self.queue)

    def pop(self):
        i = self.sample()
        self.total -= self.queue[i].proportion
        return self.queue.pop(i).object

    def __str__(self):
        return "SoftQueue(" + reduce(lambda x, y: f"{x}, {y}", self.queue) + ")"

    def probabilities(self):
        return map(lambda pair: pair.proportion / self.total, self.queue)

if __name__ == '__main__':
    main()