# -*- codeing = utf-8 #-*-
# @Time : 2022/4/2410:02 AM
# @Author :yolo
# @File :2022.4.24.1.py
# @Software ：PyCharm

import gym
env=gym.make("Taxi-v3")
observation=env.reset()
for step in range(100):
    action=agent(observation)
    observation,reward,done,info=env.step(action)
env.close()