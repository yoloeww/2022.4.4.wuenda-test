# -*- codeing = utf-8 #-*-
# @Time : 2022/4/1010:32 PM
# @Author :yolo
# @File :2022.4.10.神经网络.py
# @Software ：PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
data = loadmat('/home/kesci/input/andrew_ml_ex33507/ex3data1.mat')
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y
