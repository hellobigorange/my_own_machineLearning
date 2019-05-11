import numpy as np

def polyFeatures(X1, p):
    for i in range(2, p+1):
        X1 = np.hstack((X1, (X1[:, 0].reshape(X1.shape[0],1))**i))
    return X1