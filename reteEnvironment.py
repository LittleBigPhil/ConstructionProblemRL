from policy import SoftQueuePolicy
from reteNodes import *

class ReteEnvironment:
    def __init__(self, problem, studentPolicy = SoftQueuePolicy(), teacherPolicy = SoftQueuePolicy()):
        self.problem = problem

        self.goal = None
        self.rootNode = BasicNode()
        self.goalSelectionNode = buildProductionNode("goal", lambda arg: self.selectGoal(arg))
        self.instantiationsNode = BasicNode()

        self.problem.setupRete(self.rootNode, self.instantiationsNode)
        self.goalSelectionNode.link(self.instantiationsNode)

        self.studentPolicy = studentPolicy
        self.teacherPolicy = teacherPolicy
        self.policy = self.studentPolicy
        self.instantiationsNode.onAdd = lambda instantiation: \
            self.policy.add(instantiation, self.problem.extractFeatures(instantiation, self.goal))

        self.log = []

        self.reset()

    def step(self):
        instantiation = self.policy.resolve()
        features = self.problem.extractFeatures(instantiation, self.goal)
        inferred = instantiation.resolve()
        if inferred is not None:
            self.rootNode.add(inferred)
            self.log.append((str(instantiation), inferred, -1))
        done = self.goal in self.rootNode.objects
        return done, inferred, features

    def clear(self):
        self.rootNode.clear()
        self.rootNode.unlink(self.goalSelectionNode)
        self.policy.clear()
        self.log = []
    def reset(self):
        self.goal = None
        self.clear()

        initialObjects = self.problem.generateInitialObjects()
        self.policy = self.teacherPolicy
        self.rootNode.addMany(initialObjects)

        i = 0
        while self.goal is None:
            if i == self.problem.goalSelectionDelay():
                self.rootNode.link(self.goalSelectionNode)
            self.step()
            i += 1
            if i == self.problem.goalSelectionDelay() + 5:
                self.forceGoalSelection()
                # force a goal selection
                #print("Failed to pick a goal.")
                #break

        self.clear()
        self.policy = self.studentPolicy
        self.rootNode.addMany(initialObjects)
    def selectGoal(self, goal):
        self.goal = goal
    def forceGoalSelection(self):
        self.policy.clear()
        for instantiation in self.goalSelectionNode.objects:
            self.policy.add(instantiation, self.problem.extractFeatures(instantiation, self.goal))