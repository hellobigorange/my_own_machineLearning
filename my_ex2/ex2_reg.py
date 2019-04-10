import numpy as np
import matplotlib .pyplot as plt
from plotData import *
# from sigmod import *
from cost_function import *
import scipy.optimize as opt
from predict import *
#from gradient_desent import *
from map_feature import *
# 读取数据
print('读取数据中....')
f = open('data\ex2data2.txt')
dataset = np.loadtxt(f, delimiter=',', usecols=(0, 1, 2),  unpack=True)
Exam1_score = dataset[0]
Exam2_score = dataset[1]
admmitor = dataset[2]
map_feature(Exam1_score, Exam2_score, 6)

# 绘制散点图
A_1, B_1, A_2, B_2 = my_Plot(dataset, admmitor, Exam1_score, Exam2_score)
plt.figure(0)
plt.scatter(A_2, B_2, c='red', marker='o', s=20)  # positive
plt.scatter(A_1, B_1, c='blue', marker='x', s=20)  #negtive
plt.xlabel('Microchips Test 1', fontsize=10)  # X轴
plt.ylabel('Microchips Test 2', fontsize=10)  # Y轴
plt.legend(['y=1', 'y=0'])  # 曲线的标签
plt.show()



