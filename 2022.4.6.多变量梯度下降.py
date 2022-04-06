# -*- codeing = utf-8 -*-
# @Time : 2022/4/6 12:40
# @Author : yolo
# @File : 2022.4.6.多变量梯度下降.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def com(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)  #72行2列X2行1列=72行1列
    return np.sum(inner)/(2*len(X))

path='2000'
data=pd.read_csv(path,header=None, names=['Population', 'bedroom','Profit'])
print(data)
#进行归一化
data = (data - data.mean()) / data.std() #本值减去平均值/标准值
print(data.head())

def gradientDescent(X, y, theta, alpha, epoch):
    """reuturn theta, cost"""

    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵(1, 2)
    parameters = int(theta.flatten().shape[1])  # 参数 θ的数量
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m

    for i in range(epoch):
        # 利用向量化一步求解
        temp = theta - (alpha / m) * (X * theta.T - y).T * X
        theta = temp
        cost[i] = com(X, y, theta)
    return theta, cost



data.insert(0, 'Ones', 1)
cols=data.shape[1] #获取列数
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
#转化为矩阵
X = np.matrix(x.values)
Y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))
#赋初值
alpha = 0.01
epoch = 1000

g2, cost = gradientDescent(X, Y, theta, alpha, epoch)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch), cost, 'b')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()