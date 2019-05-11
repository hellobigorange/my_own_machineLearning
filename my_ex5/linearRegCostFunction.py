import numpy as np


def g(z):
    h = 1./(1+np.exp(-z))
    return h

"""加上正则化的代价函数及其偏导数"""
def linearRegCostFunction(X, Y, theta, lmda):
    m = X.shape[0]
    # n = X.shape[1]
    n=X.shape[1]
    # fmin输出的theta是(n,)要正确运算，需reshpe 成(n,1)               )
    theta = theta.reshape(n, 1)
    h = X.dot(theta)
    Y = Y.reshape(m, 1)

    # 代价函数
    J1 = (h - Y).T.dot(h - Y) / 2 / m
    J = (h - Y).T.dot(h - Y)/2/m + (theta[1:].T.dot(theta[1:]))*lmda/2/m
    # 代价函数的导数
    J_d = X.T.dot(h-Y)/m
    J_d[0] = J_d[0]
    J_d[1:] = J_d[1:] + lmda * theta[1:]/m
    # 由于fmin 输入的函数必得是（n,）的形式，故需要将J_dreshape成（n,）的形式
    J_d = J_d.reshape(J_d.size)
    return J, J_d, J1
