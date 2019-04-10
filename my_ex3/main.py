import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from display_Data import display_Data
from oneVsAll import one_vs_all
from lrCostFunction import lrCostFunction
from predictOneVsAll import predictOneVsAll
from nn_predict import nn_predict


# 导入数据
data = scipy.io.loadmat('data\ex3data1.mat')
Y = data['y']  # (5000,1)
X1 = data['X']  # (5000,401)
data1 = scipy.io.loadmat('data\ex3weights.mat')
Theta1 = data1['Theta1']  # (205, 401)
Theta2 = data1['Theta2']  # (10, 26)

# print(data)


"""Task1:可视化数据"""
# 显示图像
X2 = display_Data(X1, num=100)
plt.figure(1)
# 设置图像色彩为灰度值，指定图像坐标范围
plt.imshow(X2, cmap='gray', extent=[-1, 1, -1, 1])
plt.axis('off')
plt.title('Random Seleted Digits')

# 为X1增加一列，构成(5000, 401)的输入样本
X = np.hstack((np.ones((X1.shape[0])).reshape(X1.shape[0], 1), X1))
# theta_init = np.zeros((X.shape[1])).reshape(X.shape[1], 1)
lmda = 0.01

# ========================= 2.向量化Logistic Rgression =========================
# 测试函数lr_cost_function的功能
'''
theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((3, 5)).T/10]
y_t = np.array([1, 0, 1, 0, 1])
lmda_t = 3
cost,grad = lrCostFunction(X_t,y_t,theta_t,lmda_t)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
print('Cost:', cost)
print('Expected cost: 3.734819')
print('Gradients:', grad)
print('Expected gradients:\n[ 0.146561 -0.548558 0.724722 1.398003]')
'''
# #  返回训练好的10组theta
num_labels = 10
all_theta = one_vs_all(X, Y, num_labels, lmda)

# 看看逻辑回归5000组样本的预测准确性预测值
pred = predictOneVsAll(X, all_theta)
print('Training set accurayc:{}'.format(np.mean(pred == Y)*100))

# 看看神经网络5000组样本的预测准确性预测值
pred1 = nn_predict(Theta1, Theta2, X)
print('nnTraining set accurayc:{}'.format(np.mean(pred1 == Y)*100))

'''随机取一组样本，观察预测值与真实值'''

X_sample = np.random.choice(X1.shape[0], 1)
plt.figure(2)
# 设置图像色彩为灰度值，指定图像坐标范围
plt.imshow(X1[X_sample].reshape(20,20), cmap='gray', extent=[-1, 1, -1, 1])
plt.axis('off')
plt.title('a simple')
print('真实值', pred1[X_sample])

