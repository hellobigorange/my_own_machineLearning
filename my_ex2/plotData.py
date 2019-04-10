import matplotlib .pyplot as plt

# 绘制初始数据的散点图
A_2 = []
B_2 = []
A_1 = []
B_1 = []

def my_Plot(dataset, admmitor, Exam1_score, Exam2_score):
    for i in range(dataset.shape[1]):
        if admmitor[i] == 1:
            A_2 .append(Exam1_score[i])
            B_2 .append(Exam2_score[i])

        else:
            A_1.append(Exam1_score[i])
            B_1.append(Exam2_score[i])
    return A_1, B_1, A_2, B_2




# def plot_data(X, y):
#     plt.figure()
#
#     postive = X[y == 1]  # 分离正样本
#     negtive = X[y == 0]  # 分离负样本
#
#     plt.scatter(postive[:, 0], postive[:, 1], marker='+', c='red', label='Admitted')  # 画出正样本
#     plt.scatter(negtive[:, 0], negtive[:, 1], marker='o', c='blue', label='Not Admitted')  # 画出负样本
#

