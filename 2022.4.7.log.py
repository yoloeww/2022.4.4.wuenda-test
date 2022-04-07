# -*- codeing = utf-8 -*-
# @Time : 2022/4/7 11:55
# @Author : yolo
# @File : 2022.4.7.log.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
path='回归数据'
data=pd.read_csv(path,header=None,names=['exam1','exam2','admitted'])
print(data)
positive = data[data['admitted'].isin([1])] #1代表好
negative = data[data['admitted'].isin([0])] #2代表不好



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


data.insert(0, 'Ones', 1)

# 初始化X，y，θ
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)

# 转换X，y的类型
X = np.array(X.values)
y = np.array(y.values)

print(X.shape, theta.shape, y.shape)
print(cost(theta, X, y))


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)
print(cost(result[0], X, y))

plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not admitted')
ax.legend()
ax.set_xlabel('exam 1 Score')
ax.set_ylabel('exam 2 Score')
plt.show()