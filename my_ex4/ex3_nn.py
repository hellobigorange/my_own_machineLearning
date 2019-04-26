import numpy as np
from sigmoid import g


def ex33_nn(Theta1, Theta2, X):
    a1 = X
    a2 = g(a1.dot(Theta1.T))
    # 增加一个偏置维度
    a2 = np.hstack((np.ones((a2.shape[0])).reshape(a2.shape[0], 1), a2))
    a3 = g(a2.dot(Theta2.T))
    return a3