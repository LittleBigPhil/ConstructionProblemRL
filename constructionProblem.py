"""Contains the definitions of the classes which specify the problem-specific functions."""

import torch

from reteNodes import *
import numpy as np

class ConstructionProblem:
    """The abstract base class of all construction problems."""
    def generateInitialObjects(self):
        """Generates initial objects for a new task."""
        raise NotImplementedError()
    def setupRete(self, rootNode, instantiationsNode):
        """Creates the nodes and connects them to the input and output nodes."""
        raise NotImplementedError()
    def objectFeatureAmount(self) -> int:
        """The maximum size of the feature array of an object in the problem."""
        raise NotImplementedError()
    def instantiationFeatureAmount(self) -> int:
        """The maximum size of the feature array of an instantiation in the problem."""
        return self.objectFeatureAmount() * 3

    def goalSelectionDelay(self) -> int:
        """How many steps to wait before introducing the goal selection productions."""
        return 5

    def extractFeatures(self, action, goal):
        """Returns the feature representation of an action and a goal."""
        # Zero pad the feature array, because the network needs a consistent input size.
        features = np.zeros(self.instantiationFeatureAmount(), dtype=np.float32)
        try:
            for i, obj in enumerate(action.arg):
                features[i] = obj
        except:
            features[0] = action.arg

        if goal is None:
            goal = 0
        features[-1] = goal
        features = torch.from_numpy(features)
        return features

class VectorAddition(ConstructionProblem):
    """The specification for the vector addition problem."""
    def generateInitialObjects(self):
        return 1.0, 2.0, -1.0
    def setupRete(self, rootNode, instantiationsNode):
        twoVectors = ProductNode()
        addVectors = buildProductionNode("add", lambda arg: arg[0] + arg[1])

        rootNode.link(twoVectors.left)
        rootNode.link(twoVectors.right)
        twoVectors.link(addVectors)
        addVectors.link(instantiationsNode)
    def objectFeatureAmount(self):
        return 1
