import torch
import numpy as np
from constructionProblem import VectorAddition
from network import *
from policy import SoftQueuePolicy
from reteEnvironment import ReteEnvironment
from softQueue import PopInfo
import matplotlib.pyplot as plt

class ReinforcementTrainer:
    pass

def rollout(env: ReteEnvironment, policy: TrainableNetwork, critic: TrainableNetwork, maxStepsPerEpisode: int):
    """Generates raw experiences by stepping according to the policy."""
    experienceStack = []
    #print(f"goal({env.goal})")
    for i in range(maxStepsPerEpisode):
        done, inferred, features, popInfo = env.step()
        experienceStack.append((features, popInfo.entropy))
        if done:
            break
        #policyGradientUpdate(policy, critic, features, popInfo)
    #print(env.rootNode.objects)
    #print(f"\tsteps = {i+1}")
    #print(experienceStack)
    return experienceStack

def policyGradientUpdate(policy: TrainableNetwork, critic: TrainableNetwork, features, popInfo: PopInfo):
    evaluation = critic(features) - 1 # not sure if the -1 is appropriate
    factor = evaluation / popInfo.total
    logP = torch.log(torch.Tensor([popInfo.probability]))
    loss = torch.mul(logP, factor)
    policy.trainByGradient(features, loss)

def makeExperiences(experienceStack):
    """Propagates the reward back to transform raw experiences into informed experiences."""
    """ToDo:
    Add support for non-monte-carlo return."""
    entropyWeight = .1
    discountFactor = .99
    replayBuffer = []
    reward = 0
    for i in range(len(experienceStack)):
        stackIndex = len(experienceStack) - i - 1
        features, entropy = experienceStack[stackIndex]
        reward *= discountFactor
        reward += -1 + entropyWeight * entropy
        replayBuffer.append((features, reward))
        # replayBuffer.append(round(reward, 2))
    return replayBuffer

def dream(policy, replayBuffer):
    """Trains from the already generated experiences."""
    """ToDo:
    Create minibatches from the replay buffer and train on those.
    Allow for policy gradient instead of value learning."""
    for features, reward in replayBuffer:
        #reward = np.array(reward)
        #reward = torch.from_numpy(reward)
        reward = torch.tensor([reward])
        policy.train(features, reward)

def main():
    problem = VectorAddition()
    inputSize = problem.featureAmount()

    #innerPolicy = UniformWeighter()
    innerPolicy = TrainableNetwork(inputSize, inputSize * 2, 2)
    outerPolicy = SoftQueuePolicy(policy=innerPolicy)
    critic = TrainableNetwork(inputSize, inputSize * 2, 2)
    env = ReteEnvironment(problem=VectorAddition(), studentPolicy=outerPolicy)

    maxStepsPerEpisode = 20

    yVals = []
    xVals = []
    rawQuality = 0
    qualityMomentum = .99995
    bigStep = int(1 / (1 - qualityMomentum))
    print(bigStep)
    i = 0
    while True:
        i += 1
        experienceStack = rollout(env, innerPolicy, critic, maxStepsPerEpisode)
        rawQuality = qualityMomentum * rawQuality + (1 - qualityMomentum) * (maxStepsPerEpisode - len(experienceStack))
        quality = rawQuality / (1 - qualityMomentum ** i) # Adam style bias correction

        if i > bigStep:
            xVals.append(i)
            yVals.append(quality)

        #if i % 10 == 0:
            #print(f"goal({env.goal}) took {len(experienceStack)} steps. Quality is {round(quality,2)}. i is {i}.")

        replayBuffer = makeExperiences(experienceStack)
        dream(innerPolicy, replayBuffer)
        #dream(critic, replayBuffer)
        env.reset()
        if i > bigStep * 2 and i % (bigStep * 4) == 0:
            plt.plot(xVals, yVals)
            plt.show()


if __name__ == '__main__':
    main()