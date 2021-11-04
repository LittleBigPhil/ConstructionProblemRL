from environment.constructionProblem import ConstructionProblem
from environment.reteNodes import BasicNode, buildProductionNode
from learning.policy import SoftQueuePolicy

class ReteEnvironment:
    def __init__(self, problem: ConstructionProblem, studentPolicy: SoftQueuePolicy = SoftQueuePolicy(), teacherPolicy: SoftQueuePolicy = SoftQueuePolicy()):
        self.problem = problem

        self.__selectGoal(None)
        self.__setupRete()

        self.studentPolicy = studentPolicy
        self.teacherPolicy = teacherPolicy
        self.policy = self.studentPolicy

        self.reset()
    def __setupRete(self):
        self.rootNode = BasicNode()
        self.goalSelectionNode = buildProductionNode("goal", lambda arg: self.__selectGoal(arg))
        self.instantiationsNode = BasicNode()

        self.problem.setupRete(self.rootNode, self.instantiationsNode)
        self.goalSelectionNode.link(self.instantiationsNode)

        self.instantiationsNode.onAdd = lambda instantiation: \
            self.policy.add(instantiation, self.problem.extractFeatures(instantiation, self.goal))


    def step(self, withInfo=True):
        """
        Instantiates a single production according to the policy.
        @return: (done, inferred, features)
        @return done: True if the goal has been reached.
        @return actionInfo:
        @return stateInfo:
        """
        actionInfo = self.policy.resolve()
        instantiation = actionInfo.action

        inferred = instantiation.resolve()
        if inferred is not None:
            self.rootNode.add(inferred)
            self.log.append((str(instantiation), inferred, -1))
        done = self.goal in self.rootNode.objects

        if withInfo:
            def features(action):
                return self.problem.extractFeatures(action, self.goal) # Might need to capture the goal ahead of resolving
            actionInfo.action = features(instantiation)
            stateInfo = self.policy.state()  # Could skip this when "done" if need to.
            stateInfo.actionSpace = map(features, stateInfo.actionSpace)
            return done, actionInfo, stateInfo
        else:
            return done

    def __clear(self):
        """Empty everything and disable goal selection."""
        self.rootNode.clear()
        self.rootNode.unlink(self.goalSelectionNode)
        self.policy.clear()
        self.log = []
    def __manualReset(self, policy, initialObjects, resetGoal):
        """Resets the state of the environment and sets up a specific task."""
        if resetGoal:
            self.__selectGoal(None)

        self.__clear()
        self.policy = policy
        self.rootNode.addMany(initialObjects)
    def reset(self):
        """Resets the state of the environment and generates a task."""
        initialObjects = self.problem.generateInitialObjects()
        self.__manualReset(self.teacherPolicy, initialObjects, resetGoal = True)
        self.__stepUntilGoal()
        self.__manualReset(self.studentPolicy, initialObjects, resetGoal = False)

    def __selectGoal(self, goal):
        self.goal = goal
    def __forceGoalSelection(self):
        """Remove all instantiations which aren't selecting a goal.
        This forces the next action to be a goal selection."""
        self.policy.clear()
        for instantiation in self.goalSelectionNode.objects:
            self.policy.add(instantiation, self.problem.extractFeatures(instantiation, self.goal))
    def __stepUntilGoal(self):
        """Steps repeatedly until a goal has been selected."""
        i = 0
        while self.goal is None:
            if i == self.problem.goalSelectionDelay():
                self.rootNode.link(self.goalSelectionNode)
            if i == self.problem.goalSelectionDelay() + 5:
                self.__forceGoalSelection()

            self.step(withInfo=False)
            i += 1