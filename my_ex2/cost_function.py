import numpy as np
from sigmod import *


def J(theta_init, X, admmitor, dataset):
    z = np.dot(X.T, theta_init.T)
    h_theta = g(z)
    A = np.ones(dataset.shape[1])
    B = -admmitor*np.log(h_theta)-(1-admmitor) * np.log(1-h_theta)
    m = len(admmitor)
    J_theta = 1/m*np.dot(A.T, B)
    A_1 = np.array([h_theta.T - admmitor, h_theta.T - admmitor, h_theta.T - admmitor])
    grad =  1 / m * np.dot(np.ones(dataset.shape[1]).T, A_1.T * X.T)
    return J_theta, grad
