import numpy as np
from linearRegCostFunction import linearRegCostFunction
from line_trainLinearReg import line_trainLinearReg


def validationCurve(theta_init, X, X_val, y, y_val):
    lmdaa = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(lmdaa.size)
    error_val = np.zeros(lmdaa.size)
    for i in range(lmdaa.size):
        lmd = lmdaa[i]
        theta = line_trainLinearReg(theta_init, X, y, lmd)
        error_train[i],_ ,_ = linearRegCostFunction(X, y, theta, lmd)
        error_val[i],_ ,_ = linearRegCostFunction(X_val, y_val, theta, lmd)
    return error_train, error_val
