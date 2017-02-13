import gym
import time
import RLAgent as agent

gamma = 0.8
learningRate = 0.01



environment = gym.make("CartPole-v0")
observation, reward, done, info = environment.reset()
print(environment.action_space)
agent.feedForward([1, 1, 1, 2], 2)
for i in range(200):
    action = agent.learn(observation, reward)
    observation, reward, done, info = environment.step(action)
    environment.render()
    if done:
        #restart the environment
        observation, reward, done, info = environment.reset()
    else:
        action = agent.learn(observation, reward)
