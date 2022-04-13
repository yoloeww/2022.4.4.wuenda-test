# -*- codeing = utf-8 #-*-
# @Time : 2022/4/1311:11 AM
# @Author :yolo
# @File :2022.4.13.practice.线性回归.py
# @Software ：PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#总体思路，先读取数据，然后进行代价函数的定义，分别是最小二乘法使得函数值最小
path='test1'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
print(data)
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()
#代价函数
def computeCost(X, y, theta):
    inner=np.power(((X*theta.T)-y),2)  #92x2 * 2x1 =92x1
    return np.sum(inner)/(2*len(X))
data.insert(0, 'Ones', 1)
clos=data.shape[1]
X=data.iloc[:,0:clos-1]
y=data.iloc[:,clos-1:clos]
#转化为numpy矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
#一行二列的theta
theta = np.matrix([0,0])
print(theta.shape)
# (1, 2)
print(computeCost(X, y, theta)) # 32.072733877455676


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
        cost[i] = computeCost(X, y, theta)

    return theta, cost

alpha = 0.01
epoch = 1000

final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
computeCost(X, y, final_theta)

x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


