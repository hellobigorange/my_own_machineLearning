import numpy as np
import matplotlib .pyplot as plt
from plotData import *
# from sigmod import *
from cost_function import *
import scipy.optimize as opt
from predict import *
#from gradient_desent import *

print('读取数据中....')
f = open('data\ex2data1.txt')
dataset = np.loadtxt(f, delimiter=',', usecols=(0, 1, 2),  unpack=True)
Exam1_score = dataset[0]
Exam2_score = dataset[1]
admmitor = dataset[2]



# 初始化
alpha = 0.001
theta_init = np.zeros(dataset.shape[0])
X = np.array([np.ones(dataset.shape[1]), Exam1_score, Exam2_score]) # 输入样本
# z = np.dot(X.T, theta_init)  # X.T * theta
N_init = 500   #  迭代次数




# 计算初始的的J_theta和theta
#theta, J_theta = gradient_desent(N_init, X, dataset, admmitor, theta_init, alpha)
J_theta, grad = J(theta_init, X, admmitor, dataset)
print(J_theta, grad)


# # 使用非0的theta测试
# test_theta = np.array([-24,0.2,0.2])
# J_theta, grad = J(test_theta , X, admmitor, dataset)
# print(J_theta, grad)
# print('Expected cost (approx): 0.218')
# print('Expected gradients (approx): \n0.043\n2.566\n2.647')

# 使用最优化解法求J(.)和theta
def cost_func(t):
    return J(t, X, admmitor, dataset)[0]

def grad_func(t):
    return J(t, X, admmitor, dataset)[1]
# 使用opt.fmin_bfgs()来获得最优解
theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=theta_init, maxiter=400, full_output=True, disp=False)
print('Cost at theta found by fmin: {:0.3f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

# 预测新样本是否被admmited
X_test = np.array([1, 45, 85])
h_probility = predict(X_test, theta)
print(h_probility)

# 绘制分界线和散点图
A_1, B_1, A_2, B_2 = my_Plot(dataset, admmitor, Exam1_score, Exam2_score)
a=[0,-theta[0]/theta[1]]
b=[-theta[0]/theta[1],0]
plt.figure(0)
plt.plot(a,b,'g-',lw=1)  #绘制直线图
plt.scatter(A_2, B_2, c='red', marker='o', s=20)
plt.scatter(A_1, B_1, c='blue', marker='x', s=20)
plt.xlabel('Exam1_score', fontsize=10)  # X轴
plt.ylabel('Exam2_score', fontsize=10)  # Y轴
plt.axis([25,105,25,105])
plt.legend(['boundary_line','Admitted', 'not_Admitted'])  # 曲线的标签
plt.show()





# 梯度下降法绘制J_theta
# plt.figure()
# plt.plot(np.arange(J_theta.size), J_theta)
# plt.xlabel('Number of iterations')
# plt.ylabel('Cost J')
# #plt.axis([0,num_iters,0,100])
# plt.show()

