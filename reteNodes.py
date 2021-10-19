"""Contains the class definitions for the building blocks of a Rete network."""

# Helper Functions
def always(x):
    return True
def identity(x):
    return x
def doNothing1(x):
    pass
def doNothing0():
    pass

class BasicNode:
    """The base node that all nodes in a Rete network derive from."""
    def __init__(self, predicate = always, transform = identity, onAdd = doNothing1, onClear = doNothing0):
        """
        Creates a node that by default just passes through all objects.
        @param predicate: an object added to a node must pass the predicate to go through
        @param transform: an object added to a node is first transformed before being added or going through the predicate
        @param onAdd: after adding an object to a node, onAdd is called on that object
        @param onClear: after emptying the cache of the node, and clearing the children, onClear is called
        """
        self.predicate = predicate
        self.transform = transform
        self.onAdd = onAdd
        self.onClear = onClear

        self.objects = [] # The cache of added objects. This is used to avoid propagating duplicate objects.
        self.children = [] # The nodes which take the output of the current node.

    def add(self, object):
        """Applies the given functions to process a new object, adds it to the cache, and propagates it to child nodes accordingly."""
        object = self.transform(object)
        if self.predicate(object) and object not in self.objects:
            self.objects.append(object)
            self.onAdd(object)
            for child in self.children:
                child.add(object)
    def addMany(self, objects):
        """Adds a list of objects."""
        for object in objects:
            self.add(object)
    def clear(self):
        """Empties the object cache and the cache of all child nodes."""
        self.objects = []
        for child in self.children:
            child.clear()
        self.onClear()

    def link(self, childNode):
        """Connects a new child node to the current node.
        Any objects in the cache are immediately propagated to this child node."""
        self.children.append(childNode)
        for object in self.objects:
            childNode.add(object)
    def unlink(self, childNode):
        """Disconnects a child node from the current node.
        Any objects that were propagated to the child node remain in the child node."""
        if childNode in self.children:
            self.children.remove(childNode)

class ProductNode(BasicNode):
    """A node that has two parents, and combines their outputs into the cartesian product of those outputs."""
    def __init__(self):
        # Consider adding the optional parameters of BasicNode to this node.
        BasicNode.__init__(self)
        self.left = BasicNode(onAdd=self.addLeft, onClear=self.clear)
        self.right = BasicNode(onAdd=self.addRight, onClear=self.clear)

    def addRight(self, rightObject):
        """Add all the new pairs generated from a new right object."""
        newPairs = map(lambda leftObject: (leftObject, rightObject), self.left.objects)
        self.addMany(newPairs)
    def addLeft(self, leftObject):
        """Add all the new pairs generated from a new left object."""
        newPairs = map(lambda rightObject: (leftObject, rightObject), self.right.objects)
        self.addMany(newPairs)

class Instantiation:
    def __init__(self, name, arg, func):
        self.name = name
        self.arg = arg
        self.func = func

    def resolve(self):
        return self.func(self.arg)

    def __str__(self):
        return f"{self.name}{self.arg}"

def buildProductionNode(name, func):
    """Creates a node that constructs Instantiation objects corresponding to applying a function."""
    return BasicNode(transform = lambda arg: Instantiation(name, arg, func))








