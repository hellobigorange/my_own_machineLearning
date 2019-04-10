import numpy as np
from lrCostFunction import g


def ex3_nn(Theta1, Theta2, X):
    O_1 = g(X.dot(Theta1.T))
    # 增加一个偏置维度
    O_1 = np.hstack((np.ones((O_1.shape[0])).reshape(O_1.shape[0], 1), O_1))
    O_2 = g(O_1.dot(Theta2.T))
    return O_2