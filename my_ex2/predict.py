from sigmod import *
import numpy as np

def predict(X_test, theta):
    z = np.dot(X_test.T, theta)
    h_probility = g(z)
    if h_probility >= 0.5:
        print('This student is admitted')
    else:
        print('This student is not admitted')
    return h_probility