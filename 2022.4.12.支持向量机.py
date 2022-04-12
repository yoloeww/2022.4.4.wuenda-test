# -*- codeing = utf-8 #-*-
# @Time : 2022/4/1211:16 PM
# @Author :yolo
# @File :2022.4.12.支持向量机.py
# @Software ：PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

raw_data = loadmat('/home/kesci/input/andrew_ml_ex67101/ex6data1.mat')
data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')

data.head()