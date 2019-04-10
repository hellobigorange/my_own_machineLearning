from lrCostFunction import g
import numpy as np


def predictOneVsAll(X_test, all_theta):
    h = g(X_test.dot(all_theta))
    y_predict = np.argmax(h, axis=1).reshape(X_test.shape[0],1)
    return y_predict+1

# def predictOneVsAll(X, all_theta):
#     m = X.shape[0]
#     X = np.c_[np.ones(m), X]
#     # 标签数10
#     num_labels = all_theta.shape[0]
#
#     # preds[m][k]是第m个样本属于k的概率
#     preds = g(X.dot(all_theta.T))
#     P = np.zeros(m)
#
#     for num in range(m):
#         # 找到第num行中，与该行最大值相等的列的下标，此时下标的范围是[0,9]
#         # label的范围是[1,10]，需要把下标的值+1
#         # np.where()返回的是一个长度为2的元祖，保存的是满足条件的下标
#         # 元组中第一个元素保存的是行下标，第二元素保存的是列下标
#         index = np.where(preds[num, :] == np.max(preds[num, :]))
#         P[num] = index[0][0].astype(int) + 1
#
#     return P
