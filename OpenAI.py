import gym
import time

environment = gym.make("CartPole-v0")
#environment.reset

for i in range(20):
    environment.render()
    observation, reward, done, info = environment.step(0)
    time.sleep(1)

    if done:
        environment.reset()

    environment.render()
    observation, reward, done, info = environment.step(1)
    time.sleep(1)