import scipy.optimize as opt
import numpy as np

from lrCostFunction import lrCostFunction


def one_vs_all(X, Y, num_labels, lmda):
	# 特征数
	n = X.shape[1]
	# 几类分就训练几组theta
	all_theta = np.zeros((n, num_labels))
	# Y中的值是1~10
	for i in range(1, num_labels+1):
		# Y中等于i的令它为1，其余为0
		theta_init = np.zeros((n, 1))
		y = (Y == i).astype(int)

		def cost_func(t):
			return lrCostFunction(X, y, t, lmda)[0]

		def grad_func(t):
			return lrCostFunction(X, y, t, lmda)[1]

		# 使用opt.fmin_bfgs()来获得最优解
		theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=theta_init, maxiter=400, full_output=True, disp=False)
		#theta = theta.reshape(n, 1)
		all_theta[:, i-1] = theta
	return all_theta

