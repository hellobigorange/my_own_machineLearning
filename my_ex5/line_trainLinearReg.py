import scipy.optimize as opt
from linearRegCostFunction import linearRegCostFunction


def line_trainLinearReg(theta_init, X, y, lmda):
    def cost_func(t):
        return linearRegCostFunction(X, y, t, lmda)[0]

    def grad_func(t):
        return linearRegCostFunction(X, y, t, lmda)[1]

    # 使用opt.fmin_bfgs()来获得最优解
    theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=theta_init, maxiter=400, full_output=True, disp=False)
    return theta

