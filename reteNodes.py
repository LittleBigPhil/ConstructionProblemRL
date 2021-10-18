
def always(x):
    return True
def identity(x):
    return x
def doNothing1(x):
    pass
def doNothing0():
    pass

class BasicNode:
    def __init__(self, predicate = always, transform = identity, onAdd = doNothing1, onClear = doNothing0):
        self.predicate = predicate
        self.transform = transform
        self.onAdd = onAdd
        self.onClear = onClear
        self.objects = []
        self.children = []

    def add(self, object):
        object = self.transform(object)
        if self.predicate(object) and object not in self.objects:
            self.objects.append(object)
            self.onAdd(object)
            for child in self.children:
                child.add(object)
    def addMany(self, objects):
        for object in objects:
            self.add(object)
    def clear(self):
        self.objects = []
        for child in self.children:
            child.clear()
        self.onClear()

    def link(self, childNode):
        self.children.append(childNode)
        for object in self.objects:
            childNode.add(object)
    def unlink(self, childNode):
        if childNode in self.children:
            self.children.remove(childNode)

class ProductNode(BasicNode):
    def __init__(self):
        BasicNode.__init__(self)
        self.left = BasicNode(onAdd=self.addLeft, onClear=self.clear)
        self.right = BasicNode(onAdd=self.addRight, onClear=self.clear)

    def addRight(self, rightObject):
        newPairs = map(lambda leftObject: (leftObject, rightObject), self.left.objects)
        self.addMany(newPairs)
    def addLeft(self, leftObject):
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
    return BasicNode(transform = lambda arg: Instantiation(name, arg, func))








