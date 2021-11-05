import matplotlib.pyplot as plt
import learning.algorithms as algorithms

from configLoader import Configuration
from learning.training import ReinforcementTrainer
from reporting.momentum import BiasCorrectedMomentum

def main(maxI: int = -1):
    rlTrainer = ReinforcementTrainer(algorithms.SoftQ())
    maxStepsPerEpisode = Configuration.load().maxStepsPerEpisode

    yVals = []
    xVals = []

    quality = BiasCorrectedMomentum(Configuration.load().qualityMomentum)
    bigStep = Configuration.load().episodesPerTimeStep

    i = 0
    while maxI < 0 or i < maxI:
        i += 1
        length = rlTrainer.rollout(maxStepsPerEpisode)

        quality.add(maxStepsPerEpisode - length)
        if i > quality.timeScale() and i % 100 == 0:
            xVals.append(i)
            yVals.append(quality.get())
        if i > 0 and i % bigStep == 0:
            plt.plot(xVals, yVals)
            plt.show()


if __name__ == '__main__':
    main()