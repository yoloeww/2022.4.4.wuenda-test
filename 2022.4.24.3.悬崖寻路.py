# -*- codeing = utf-8 #-*-
# @Time : 2022/4/244:47 PM
# @Author :yolo
# @File :2022.4.24.3.悬崖寻路.py
# @Software ：PyCharm

import gym
import math
import numpy as np

import numpy as np

class Qlearning():
    def choose_action(self,state):
        self.sample_count+=1
        self.epsilon=self.epsilon_end+(self.epsilon_start-self.epsilon_end)*math.exp(-1.*self.sample_count/self.epsilon_decay)
        if np.random.uniform(0,1)>self.epsilon:
            action=np.argmax(self.Q_table[str(state)])
        else:
            action=np.random.choice(self.action_dim)
        return action
env = gym.make("CliffWalking-v0") # 0 up, 1 right, 2 down, 3 left
env.seed(1)
n_states= env.observation_space.n
n_actions=env.action_space.n
print(f"{n_states}",f"{n_actions}")
agent=Qlearning(n_states,n_actions,cfg)
for i_ep in range(cfg.train_eps):
    ep_reward=0
    state=env.reset()
    while True:
        action=agent.choose_action(state)
        next_state,reward,done,_=env.step(action)
        agent.update(state,action,reward,next_state,done)
        state=next_state
        ep_reward+=reward
        if done:
            break

rewards=[]
ma_rewards=[]

for i_ep in range(cfg.train_eps):
    ep_reward=0
    state=env.reset()
    while True:
        action=agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        if done:
            break
rewards.append(ep_reward)
if ma_rewards:
    ma_rewards.append(ma_rewards[-1]*0.0+ep_reward*0.1)
else:
    ma_rewards.append(ep_reward)



def update(self,action,reward,next_state,done):
    Q_predict=self.Q_table[str(state)][action]
    if done:
        Q_target=reward
    else:
        Q_target=reward+self.gamma
    self.Q_table[str(state)][action]+=self.lr*(Q_target-Q_predict)





