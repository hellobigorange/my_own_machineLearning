import numpy as np


def g(z):
    g = 1/(1+np.exp(-z))
    return g


def g_d(z):
    g_d = g(z)*(1-g(z))
    return g_d


def ex3_nn(X1, Y, nn_parameters, K, X, layer, lmda):
    n = X1.shape[1]
    Theta1 = nn_parameters[:layer * (n + 1)].reshape(layer, n + 1)
    Theta2 = nn_parameters[layer * (n + 1):].reshape(K, layer + 1)
    # np.set_printoptions(precision=20)
    '''一、前向传播计算'''
    #a1 = X  # (5000,401)
    z2 = X.dot(Theta1.T)  # (5000,25)
    a2 = g(z2)  # (5000, 25)
    # 增加一个偏置维度
    a2 = np.hstack((np.ones((a2.shape[0])).reshape(a2.shape[0], 1), a2)) #(5000, 26)
    z3 = a2.dot(Theta2.T)  # (5000, 10)
    a3 = g(z3)
    '''二、计算代价'''
    m = X1.shape[0]

    # 把Y变成[00100..]的形式，输出为Z
    Z = np.zeros((m, K))
    for i in range(m):
        Z[i][Y[i]-1] = 1
    # 计算J_theta
    h = a3.reshape(m*K, 1)
    Y1 = Z.reshape(m*K, 1)
    Theta11 = Theta1[:, 1:].reshape(n*layer, 1)
    Theta22 = Theta2[:, 1:].reshape(K*layer, 1)
    J = (-Y1.T.dot(np.log(h)) - (1 - Y1).T.dot(np.log(1 - h))) / m + (Theta11.T.dot(Theta11)+ Theta22.T.dot(Theta22))*lmda/2/m
    '''三、反向传播计算梯度'''
    delta3 = a3 - Z  # (5000,10)
    z2 = np.hstack((np.ones((z2.shape[0])).reshape(z2.shape[0], 1), z2)) # (5000,26)
    delta2 = delta3.dot(Theta2)*g_d(z2)  # (5000,10)*(10,26)*(5000, 26)
    delta2 = delta2[:, 1:]  # (5000,25)
    # 反向传播计算计算梯度
    theta1_d = np.zeros(Theta1.shape)  # (25, 401)
    theta1_d = (theta1_d + delta2.T.dot(X))/m+ lmda/m*(np.column_stack((np.zeros(Theta1.shape[0]),Theta1[:,1:])))
    theta2_d = np.zeros(Theta2.shape)  # (10, 26)
    theta2_d = (theta2_d + delta3.T.dot(a2))/m + lmda/m*(np.column_stack((np.zeros(Theta2.shape[0]),Theta2[:,1:])))

    # 返回梯
    grad = np.concatenate([theta1_d.flatten(), theta2_d.flatten()])
    return J, grad