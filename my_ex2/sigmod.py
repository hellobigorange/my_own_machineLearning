import numpy as np


def g(z):
    g_sigmod_h = 1./(1+np.exp(-z))
    return g_sigmod_h