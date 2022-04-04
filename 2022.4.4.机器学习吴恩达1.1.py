# -*- codeing = utf-8 -*-
# @Time : 2022/4/4 22:49
# @Author : yolo
# @File : 2022.4.4.机器学习吴恩达1.1.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path =  'ex1data1'
# names添加列名，header用指定的行来作为标题，若原无标题且指定标题则设为None
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# 前几行print(data.head())
# 方差 标准值 print(data.describe())
print(data)
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
plt.show()
