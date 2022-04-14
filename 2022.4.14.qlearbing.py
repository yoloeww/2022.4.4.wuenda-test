# -*- codeing = utf-8 #-*-
# @Time : 2022/4/143:49 PM
# @Author :yolo
# @File :2022.4.14.qlearbing.py
# @Software ：PyCharm
import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES=6
actions=['left','right']
epsilon=0.9  #选择动作
alpha=0.1#学习效率
LAMDBDA=0.9#reward
max=13 #回合
time=0.3 #走一步花费时间

def build_q_table(states,actions):
    table=pd.dataframe(
        np.zeros((states,len(actions))),
        columns=actions,
    )
    print(table)
    return table

def choose_action(state,q_table):


