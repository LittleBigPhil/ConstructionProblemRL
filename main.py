from reteEnvironment import ReteEnvironment
from constructionProblem import *

def main():
    env = ReteEnvironment(problem = VectorAddition())
    print(f"goal({env.goal})")
    for i in range(100):
        done, inferred = env.step()
        #print(inferred)
        if done:
            break
    print(env.rootNode.objects)
    print(len(env.rootNode.objects) - 3)
    print(env.policy.entropy())
    print(env.log)

if __name__ == '__main__':
    main()
