import matplotlib.pyplot as plt

from configLoader import *

class BiasCorrectedMomentum:
    """Keeps track of a statistic that is updated according to momentum and has Adam-style bias correction."""
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.rawValue = 0
        self.i = 0

    def add(self, value: float) -> None:
        """Updates the statistic according to the most recently seen value."""
        self.rawValue = self.momentum * self.rawValue + (1 - self.momentum) * value
        self.i += 1
    def get(self) -> float:
        """Returns the statistic."""
        return self.rawValue / (1 - self.momentum ** self.i) # Adam style bias correction

    def timeScale(self) -> int:
        """
        Returns the equivalent of the denominator for a weighted average.
        .99 -> 100, .999 -> 1000
        """
        return int(1 / (1 - self.momentum))


def main():
    ys = []
    #s = BiasCorrectedMomentum(.99)
    s = BiasCorrectedMomentum(Configuration.load().qualityMomentum)
    for i in range(100):
        s.add(i % 2)
        ys.append(s.get())
    plt.plot(ys)
    plt.show()

if __name__ == "__main__":
    main()