# -*- codeing = utf-8 #-*-
# @Time : 2022/4/2410:12 AM
# @Author :yolo
# @File :2022.4.24.2.py
# @Software ：PyCharm

import gym
env=gym.make("MountainCar-v0")
print('观测空间={}'.format(env.observation_space))
print('动作空间={}'.format(env.action_space))
print('观测范围={}~{}'.format(env.observation_space.low,env.observation_space.high))
print('动作数={}'.format(env.action_space.n))
class BespokeAgent:
    def __init__(self,env):
        pass
    def decide(self,observation):
        position,velocity=observation
        lb=min(-0.09*(position+0.25)**2+0.03+0.03,0.3*(position+0.9)**4-0.008)
        ub=-0.07*(position+0.38)**2+0.07
        if lb<velocity<ub:
            action=2
        else:
            action=0
        return action
    def learn(self,*args):
        pass

agent=BespokeAgent(env)

def play_montecarlo(env,agent,render=False,train=False):
    episode_reward=0.
    observation=env.reset()
    while True:
        if render:
            env.render()
        action=agent.decide(observation)
        next_observation,reward,done,_=env.step(action)
        episode_reward+=reward
        if train:
            agent.learn(observation,action,reward,done)
        if done:
            break
        observation=next_observation
    return episode_reward

env.seed(4)
episode_reward=play_montecarlo(env,agent,render=True)
print('回合奖励={}'.format(episode_reward))
env.close()