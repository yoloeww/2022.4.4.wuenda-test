# -*- codeing = utf-8 #-*-
# @Time : 2022/4/97:36 PM
# @Author :yolo
# @File :2022.4.9.信息论.py
# @Software ：PyCharm
import numpy as np
import scipy.stats
# 交叉熵计算
def cross(m1,n1):
    return -np.sum(m1*np.log(n1)+(1-m1)*np.log(1-n1))#m出现的概率乘以n信息量

m2=np.array([0.8,0.3,0.12,0.04],dtype=float)
n2=np.array([0.8,0.3,0.12,0.04],dtype=float)
m=np.array([0.8,0.3,0.12,0.04],dtype=float)
n=np.array([0.7,0.28,0.08,0.06],dtype=float)
print("交叉shang",cross(m,n))

def kl_D(m,n):
    return scipy.stats.entropy(m,n)
m=np.array([0.8,0.3,0.12,0.04],dtype=float)
n=np.array([0.7,0.28,0.08,0.06],dtype=float)
m2=np.array([0.8,0.3,0.12,0.04],dtype=float) #越小差异越小
n2=np.array([0.8,0.3,0.12,0.04],dtype=float)
print("差异shang",kl_D(m2,n2))


#交叉熵用来形容，从事件m的角度来衡量n，如何描述n,适用于衡量不同事件n的差异。
