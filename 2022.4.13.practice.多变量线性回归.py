# -*- codeing = utf-8 #-*-
# @Time : 2022/4/1311:55 AM
# @Author :yolo
# @File :2022.4.13.practice.多变量线性回归.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#总体思路，先读取数据，然后进行代价函数的定义，分别是最小二乘法使得函数值最小
path='test2'
data=pd.read_csv(path,header=None,names=['Size', 'Bedrooms', 'Price'])
data = (data - data.mean()) / data.std()

data.insert=(0,'Ones',1)
clos=data.shape[1]
X=data.iloc[:,0:clos-1]
y=data.iloc[:,clos-1:clos]

X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0]))

def cost1(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
print(X.shape)


def dinger(X,y,theta,a,epoch):

    temp=np.matrix(np.zeros(theta.shape))
    para=int(theta.flatten().shape[1])
    cost=np.zeros(epoch)
    m=X.shape[1]

    for i in range(epoch):
        temp=theta-(a/m)*(X*theta.T-y).T*X
        theta=temp
        cost[i] = cost1(X, y, theta)

    return theta, cost
alpha = 0.01
epoch = 1000
last,cost=dinger(X, y, theta, alpha, epoch)
cost1(X, y, last)

x = np.linspace(data.Size.min(), data.Size.max(), 100)  # 横坐标
f = last[0, 0] + (last[0, 1] * x) +(last[0,2]*x) # 纵坐标，利润

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


