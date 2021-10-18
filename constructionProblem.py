import torch

from reteNodes import *
import numpy as np

class ConstructionProblem:
    def generateInitialObjects(self):
        raise NotImplementedError()
    def setupRete(self, rootNode, instantiationsNode):
        raise NotImplementedError()
    def featureAmount(self):
        raise NotImplementedError()

    def goalSelectionDelay(self):
        return 5

    def extractFeatures(self, action, goal):
        if goal is None:
            goal = 0
        features = np.append(np.array(action.arg), goal)
        features = np.pad(features, (0, self.featureAmount()-len(features)), 'constant')
        features = torch.from_numpy(features)
        return features.float()

class VectorAddition(ConstructionProblem):
    def generateInitialObjects(self):
        return 1.0, 2.0, -1.0
    def setupRete(self, rootNode, instantiationsNode):
        twoVectors = ProductNode()
        addVectors = buildProductionNode("add", lambda arg: arg[0] + arg[1])

        rootNode.link(twoVectors.left)
        rootNode.link(twoVectors.right)
        twoVectors.link(addVectors)
        addVectors.link(instantiationsNode)
    def featureAmount(self):
        return 3
