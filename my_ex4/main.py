import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from display_Data import display_Data
from nn_predict import nn_predict
from nnCostFunction import ex3_nn
from nnCostFunction import g_d
from randInitializeWeights import randInitializeWeights
from checkNNGradients import check_nn_gradients
import scipy.optimize as opt

# 导入数据
# 输出层和隐藏层个数

#np.set_printoptions(precision=20)
K = 10
layer = 25

INIT_EpSILON = 0.12
data = scipy.io.loadmat('data\ex3data1.mat')
Y = data['y']  # (5000,1)
X1 = data['X']  # (5000,400)
n = X1.shape[1]
# data1 = scipy.io.loadmat('data\ex3weights.mat')
# Theta1 = data1['Theta1']  # (205, 401)
# Theta2 = data1['Theta2']  # (10, 26)
# nn_parameters = np.concatenate([Theta1.flatten(),Theta2.flatten()])
# print(data)


"""Task1:可视化数据"""
# # 显示图像
# X2 = display_Data(X1, num=100)
# plt.figure(1)
# # 设置图像色彩为灰度值，指定图像坐标范围
# plt.imshow(X2, cmap='gray', extent=[-1, 1, -1, 1])
# plt.axis('off')
# plt.title('Random Seleted Digits')
X = np.hstack((np.ones((X1.shape[0])).reshape(X1.shape[0], 1), X1))
# 看看神经网络5000组样本的预测准确性预测值
# pred1,O_2 = nn_predict(Theta1, Theta2, X)
# print('nnTraining set accurayc:{}'.format(np.mean(pred1 == Y)*100))


# '''随机取一组样本，观察预测值与真实值'''
#
# X_sample = np.random.choice(X1.shape[0], 1)
# plt.figure(2)
# # 设置图像色彩为灰度值，指定图像坐标范围
# plt.imshow(X1[X_sample].reshape(20,20), cmap='gray', extent=[-1, 1, -1, 1])
# plt.axis('off')
# plt.title('a simple')
# print('真实值', pred1[X_sample])
#

# '''检查初始代价是否正确'''
# lmda = 3
# J, grad = ex3_nn(X1, Y, nn_parameters, K, Theta1, Theta2, X, layer, lmda)
#
# print('from weights import theta1,theta2 to veritify J(0.57..):', J)
#

# """检查sigmoid函数"""
#
# z1 = 100
# z2 = 0
# g_d1 = g_d(z1)
# g_d2 = g_d(z2)
# print('z1(0)= %f\n, z2(0.25)=%f\n'%(g_d1,g_d2))


"""random_init_theta"""
Theta1 = randInitializeWeights(X1.shape[1], layer)
Theta2 = randInitializeWeights(layer, K)
rand_nn_parameters = np.concatenate([Theta1.flatten(),Theta2.flatten()])
# print(Theta1.shape)

# # 检查BP算法
#
# lmda =3
# check_nn_gradients(lmda)

# 训练网络

lmda = 1
def cost_func(t):
    return ex3_nn(X1, Y, t, K, X, layer, lmda)[0]


def grad_func(t):
    return ex3_nn(X1, Y, t, K, X, layer, lmda)[1]


# 使用opt.fmin_bfgs()来获得最优解
print('I am training...')
nn_parameter, cost, *unused = opt.fmin_cg(f=cost_func, fprime=grad_func, x0=rand_nn_parameters, maxiter=100, full_output=True, disp=True)
Theta1 = nn_parameter[:layer * (n + 1)].reshape(layer, n + 1)
Theta2 = nn_parameter[layer * (n + 1):].reshape(K, layer + 1)
# 看看神经网络5000组样本的预测准确性预测值
pred11, O_2 = nn_predict(Theta1, Theta2, X)
print('nnTraining set accurayc:{}'.format(np.mean(pred11 == Y)*100))

#可视化
X2 = display_Data(Theta1[:, 1:], num=25)
plt.figure(1)
# 设置图像色彩为灰度值，指定图像坐标范围
plt.imshow(X2, cmap='gray', extent=[-1, 1, -1, 1])
plt.axis('off')
plt.title('Random Seleted Digits')