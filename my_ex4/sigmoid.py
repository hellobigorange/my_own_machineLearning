import numpy as np
# 计算sigmoid函数值
def g(z):
	g = 1/(1+np.exp(-z))
	return g


def g_d(z):
	g_d = g(z)*(1-g(z))
	return g_d