# -*- codeing = utf-8 #-*-
# @Time : 2022/4/134:55 PM
# @Author :yolo
# @File :2022.4.17.强化学习1.py
# @Software ：PyCharm
import gym
import pygame
import tkinter
env=gym.make("CartPole-v0")
state=env.reset()
for step in range(10000):
    env.render()
    print(state)
    action=env.action_space.sample()
    state,reward,done,info=env.step(action)
    if done:
        print('end')
        break
env.close()
