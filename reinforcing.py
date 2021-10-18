import torch

from constructionProblem import VectorAddition
from network import TrainableNetwork
from policy import SoftQueuePolicy
from reteEnvironment import ReteEnvironment


class ReinforcementTrainer:
    pass

def rollout(env: ReteEnvironment):
    experienceStack = []
    #print(f"goal({env.goal})")
    for i in range(50):
        entropy = env.policy.entropy()
        done, inferred, features = env.step()
        experienceStack.append((features, entropy))
        if done:
            break
    #print(env.rootNode.objects)
    #print(f"\tsteps = {i+1}")
    #print(experienceStack)
    return experienceStack

def makeExperiences(experienceStack):
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
    for features, reward in replayBuffer:
        #reward = np.array(reward)
        #reward = torch.from_numpy(reward)
        reward = torch.tensor([reward])
        policy.train(features, reward)

def main():
    problem = VectorAddition()
    inputSize = problem.featureAmount()

    innerPolicy = TrainableNetwork(inputSize, inputSize * 2, 2)
    outerPolicy = SoftQueuePolicy(policy=innerPolicy)
    critic = TrainableNetwork(inputSize, inputSize * 2, 2)
    env = ReteEnvironment(problem=VectorAddition(), studentPolicy=outerPolicy)

    quality = 100

    for i in range(1000):
        experienceStack = rollout(env)
        quality = .99 * quality + .01 * len(experienceStack)
        print(f"goal({env.goal}) took {len(experienceStack)} steps. Quality is {round(quality,2)}.")

        replayBuffer = makeExperiences(experienceStack)
        #dream(innerPolicy, replayBuffer)
        dream(critic, replayBuffer)
        env.reset()

if __name__ == '__main__':
    main()