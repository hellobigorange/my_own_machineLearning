import numpy as np


"""显示数字函数，其中dat0a表示要显示的数组，num表示要显示的总行数"""


def display_Data(X1, num=100):
    # 随机决定5000行样本被挑选100行
    column_X1 = np.random.choice(X1.shape[0], num)
    k = 0
    j = 0
    X2 = np.zeros((200, 200))
    for i in column_X1:
        X2[20*j:20*(j+1), 20*k:20*(k+1)] = X1[i].reshape(20, 20)
        k = k + 1
        if k == 10:
            k = 0
            j = j+1
    return X2





