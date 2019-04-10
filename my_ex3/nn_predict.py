from ex3_nn import ex3_nn
import numpy as np

def nn_predict(Theta1, Theta2, X):
    O_2 = ex3_nn(Theta1, Theta2, X)
    y_predict = np.argmax(O_2, axis=1).reshape(X.shape[0], 1)
    return y_predict + 1