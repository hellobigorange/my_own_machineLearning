import numpy as np
from sigmod import *
from cost_function import *


def gradient_desent(N_init, X, dataset, admmitor, theta_init, alpha):
    theta = theta_init.T
    J_theta = np.zeros(N_init)
    for i in range(N_init):
        z = np.dot(X.T, theta)
        h_theta = g(z)
        m = len(admmitor)
        A_1 = np.array([h_theta.T - admmitor, h_theta.T - admmitor, h_theta.T - admmitor])
        theta = theta - alpha/m*np.dot(np.ones(dataset.shape[1]).T, A_1.T*X.T)
        J_theta[i] = J(z, admmitor, dataset)


    return theta, J_theta



