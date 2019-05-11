import numpy as np
from line_trainLinearReg import line_trainLinearReg
from linearRegCostFunction import linearRegCostFunction
def learningCurve(X, Xval, y, yval, theta_init, lmda):
    m = X.shape[0]
    J_test = np.zeros(m)
    J_val = np.zeros(m)
    for i in range(m):
        X_A = X[:i+1, :] # 新样本长度
        y_A = y[:i+1]
        theta = line_trainLinearReg(theta_init, X_A, y_A, lmda)
        J_test[i] = linearRegCostFunction(X_A, y_A, theta, lmda)[2]
        J_val[i] = linearRegCostFunction(Xval, yval, theta, lmda)[2]
    return J_test, J_val


# def learningCurve(X,Y,Xval,Yval,lmda,theta_init):
#     m = X.shape[0]
#     J_train = np.zeros(m)
#     J_val = np.zeros(m)
#     for num in range(m):
#         theta = line_trainLinearReg(theta_init, X[0:num+1,:], Y[0:num+1], lmda)
#         J_train[num],_ = linearRegCostFunction(X[0:num+1,:], Y[0:num+1], theta, lmda)
#         J_val[num],_ = linearRegCostFunction(Xval, Yval, theta, lmda)
#     return J_train,J_val