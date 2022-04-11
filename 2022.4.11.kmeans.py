# -*- codeing = utf-8 #-*-
# @Time : 2022/4/1110:52 PM
# @Author :yolo
# @File :2022.4.11.kmeans.py
# @Software ：PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
def findClosestCentroids(X, centroids):
    """
    output a one-dimensional array idx that holds the
    index of the closest centroid to every training example.
    """
    idx = []
    max_dist = 1000000  # 限制一下最大距离
    for i in range(len(X)):
        minus = X[i] - centroids  # here use numpy's broadcasting
        dist = minus[:,0]**2 + minus[:,1]**2
        if dist.min() < max_dist:
            ci = np.argmin(dist)
            idx.append(ci)
    return np.array(idx)
mat = loadmat('data/ex7data2.mat')
# print(mat.keys())
X = mat['X']
init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, init_centroids)
print(idx[0:3])
