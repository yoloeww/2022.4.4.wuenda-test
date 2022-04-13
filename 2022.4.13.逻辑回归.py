# -*- codeing = utf-8 #-*-
# @Time : 2022/4/1310:33 PM
# @Author :yolo
# @File :2022.4.13.逻辑回归.py
# @Software ：PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path='test3'
data=pd.read_csv('test3',header=None,name=['exam1', 'exam2', 'admitted'])