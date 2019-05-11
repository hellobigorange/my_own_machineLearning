import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from linearRegCostFunction import linearRegCostFunction
from line_trainLinearReg import line_trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNorm import feature_normalize
from plotFit import plotFit
from validationCurve import validationCurve


data = scipy.io.loadmat('data/ex5data1.mat')
# print(data.keys())
'''绘制training data'''
X1 = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval1 = data['Xval']
yval = data['yval']

'''编写代价函数'''
# 为X1增加一列，构成(12, 2)的输入样本
X = np.hstack((np.ones((X1.shape[0])).reshape(X1.shape[0], 1), X1))
Xval = np.hstack((np.ones((Xval1.shape[0])).reshape(Xval1.shape[0], 1), Xval1))
theta_init = np.ones((2, 1))
lmda = 1
J, J_d, J1 = linearRegCostFunction(X, y, theta_init, lmda)
print('预计代价应为303.993', J)
print('预计梯度应为[-15.30; 598.250]', J_d)


'''用线性函数拟合样本'''
n = X.shape[1]
m = X.shape[0]
# fmin输出的theta是(n,)要正确运算，需reshpe 成(n,1)
theta = line_trainLinearReg(theta_init, X, y, lmda)
theta = theta.reshape(n, 1)
a = [-50, 40]
b = [theta[0]-50*theta[1], theta[0]+40*theta[1]]
plt.figure(0)
plt.plot(a, b, 'b-', lw=1)  # 绘制直线图
plt.scatter(X1, y, c='red', marker='x', s=20)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

'''绘制误差曲线，观察是否过拟合欠拟合'''
J_train, J_val = learningCurve(X, Xval, y, yval, theta_init, lmda)
plt.figure(1)
plt.plot(range(m),J_train, range(m),J_val)
plt.title('Learning curve for linear regression')
plt.xlabel('m')
plt.ylabel('Error')
plt.show()


"""应用到训练集、测试集、交叉验证集"""
'''增加特征数'''
X1_pol = polyFeatures(X1, 8)
Xtest1_pol = polyFeatures(Xtest, 8)
Xval1_pol = polyFeatures(Xval1, 8)

"""特征归一化"""
X2_pol, X_pol_mean, X_pol_std = feature_normalize(X1_pol)
# X2test_pol, Xtest_pol_mean, Xtest_pol_std = feature_normalize(Xtest1_pol)
# X2val_pol, Xval_pol_mean, Xval_pol_std = feature_normalize(Xval1_pol)
X2test_pol = (Xtest1_pol- X_pol_mean)/X_pol_std
X2val_pol = (Xval1_pol- X_pol_mean)/X_pol_std

X_pol = np.hstack((np.ones((X2_pol.shape[0])).reshape(X2_pol.shape[0], 1), X2_pol))
Xtest_pol = np.hstack((np.ones((X2test_pol.shape[0])).reshape(X2test_pol.shape[0], 1), X2test_pol))
Xval_pol = np.hstack((np.ones((X2val_pol.shape[0])).reshape(X2val_pol.shape[0], 1), X2val_pol))


"""线性回归最优化学习参数theta"""
theta_pol_init = np.zeros((9, 1))
lmda = 3
theta_pol = line_trainLinearReg(theta_pol_init, X_pol, y, lmda)
x, yfit = plotFit(min(X1), max(X1), X_pol_mean, X_pol_std, theta_pol, 8)

'''用线性函数拟合样本'''
plt.figure(2)
plt.plot(x, yfit, 'b-', lw=1)  # 绘制直线图
plt.scatter(X1, y, c='red', marker='x', s=20)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

'''绘制误差曲线，观察是否过拟合欠拟合'''
J_train_pol, J_val_pol = learningCurve(X_pol, Xval_pol, y, yval, theta_pol_init, lmda)
plt.figure(3)
plt.plot(range(m),J_train_pol, range(m),J_val_pol)
plt.title('Learning curve for polynomial regression')
plt.xlabel('m')
plt.ylabel('Error')
plt.show()

"""自动选择lmda"""
lmda = np.array([0,0.001,0.003, 0.01,0.03, 0.1, 0.3, 1, 3, 10])
J_train_pol_l, J_val_pol_l = validationCurve(theta_pol_init, X_pol, Xtest_pol, y, ytest)
plt.figure(4)
plt.plot(lmda,J_train_pol_l, lmda,J_val_pol_l)
plt.title('Learning lmda for polynomial regression')
plt.xlabel('lmda')
plt.ylabel('Error')
plt.show()

lmda = 3
theta = line_trainLinearReg(theta_pol_init,  X_pol, y, lmda)
J = linearRegCostFunction(Xval_pol, yval, theta, lmda)[0]
print('J Suppose to be 3.8599', J)


