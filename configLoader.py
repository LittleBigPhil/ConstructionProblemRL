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
            with open("config.yaml", "r") as f:
                data = yaml.load(f, Loader = yaml.FullLoader)
                return Configuration(data)
        else:
            return __loaded__

    def __init__(self, data: dict):
        """The configuration details are transformed from a dictionary entry into an attribute."""
        self.hiddenLayers = data["hiddenLayers"]
        self.hiddenLayerSizeFactor = data["hiddenLayerSizeFactor"]

        self.maxStepsPerEpisode = data["maxStepsPerEpisode"]
        self.learningRate = data["learningRate"]
        self.entropyWeight = data["entropyWeight"]
        self.discountFactor = data["discountFactor"]

        self.replayBufferSize = data["replayBufferSize"]
        self.replayBatchSize = data["replayBatchSize"]

        self.qualityMomentum = data["qualityMomentum"]

if __name__ == "__main__":
    main()