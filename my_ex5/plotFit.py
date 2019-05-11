import matplotlib.pyplot as plt
import numpy as np
from polyFeatures import polyFeatures
# 输入横坐标范围，计算出拟合曲线在每一点的取值
def plotFit(min_x,max_x,mu,sigma,theta,p):
	x = np.arange(min_x-10,max_x+10, 0.05)
	x = x.reshape(x.size, 1)

	X_poly = polyFeatures(x, p)
	X_poly -= mu
	X_poly /= sigma
	X_poly = np.column_stack((np.ones(x.size), X_poly))

	Y_fit = X_poly.dot(theta)

	return x, Y_fit

