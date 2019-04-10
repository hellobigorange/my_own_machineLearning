import numpy as np


def g(z):
    h = 1./(1+np.exp(-z))
    return h

"""加上正则化的代价函数及其偏导数"""
def lrCostFunction(X, Y, theta, lmda):
    m = X.shape[0]
    n = X.shape[1]
    # fmin输出的theta是(n,)要正确运算，需reshpe 成(n,1                )
    theta = theta.reshape(n, 1)
    h = g(X.dot(theta))
    Y = Y.reshape(m,1)
    #Y = Y.reshape(Y.size)

    # 代价函数
    J = (-Y.T.dot(np.log(h))-(1-Y).T.dot(np.log(1-h)))/m+(theta.T.dot(theta))*lmda/2/m
    # J = (-Y*np.log(h)-(1-Y)*np.log(1-h)).mean()+(theta.T.dot(theta))*lmda/2/m
    # 代价函数的导数
    J_d = X.T.dot(h-Y)/m
    J_d[0]=J_d[0]
    J_d[1:] = J_d[1:] + (lmda * theta[1:]) / m
    # 由于fmin 输入的函数必得是（n,）的形式，故需要将J_dreshape成（n,）的形式
    J_d = J_d.reshape(J_d.size)
    return J, J_d
