import yaml

def main():
    maxStepsPerEpisode = Configuration.load().maxStepsPerEpisode
    print(maxStepsPerEpisode)
    """with open("config.yaml", "r") as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
        print(data)
        Configuration(data)"""

class Configuration:
    """Stores the configuration details."""

    # Singleton that only loads the configuration file once.
    __loaded__ = None
    @staticmethod
    def load() -> 'Configuration': # Expressing the type hint as a forward reference.
        if Configuration.__loaded__ is None:
            try:
                with open("config.yaml", "r") as f:
                    data = yaml.load(f, Loader = yaml.FullLoader)
                    Configuration.__loaded__ = Configuration(data)
            except IOError:
                with open("../config.yaml", "r") as f:
                    data = yaml.load(f, Loader = yaml.FullLoader)
                    Configuration.__loaded__ = Configuration(data)
        return Configuration.__loaded__

    def __init__(self, data: dict):
        """The configuration details are transformed from a dictionary entry into an attribute."""
        self.hiddenLayers = int(data["hiddenLayers"])
        self.hiddenLayerSizeFactor = data["hiddenLayerSizeFactor"]
        self.softMaxSensitivity = data["softMaxSensitivity"]
        self.softMaxOffset = data["softMaxOffset"]

        self.maxStepsPerEpisode = int(data["maxStepsPerEpisode"])
        self.learningRate = data["learningRate"]
        self.adamBeta1 = data["adamBeta1"]
        self.adamBeta2 = data["adamBeta2"]
        self.entropyWeight = data["entropyWeight"]
        self.discountFactor = data["discountFactor"]

        self.replayBufferSize = int(data["replayBufferSize"])
        self.replayBatchSize = int(data["replayBatchSize"])

        self.argMaxSampleSize = int(data["argMaxSampleSize"])
        self.swapNetworkPeriod = int(data["swapNetworkPeriod"])

        self.episodesPerTimeStep = int(data["episodesPerTimeStep"])
        self.qualityMomentum = data["qualityMomentum"]

if __name__ == "__main__":
    main()