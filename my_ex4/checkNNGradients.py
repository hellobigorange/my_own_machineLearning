import numpy as np
import randInitializeWeights as diw
import nnCostFunction as ncf
import computeNumericalGradient as cng


def check_nn_gradients(lmd):
    # np.set_printoptions(precision=20)
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generatesome 'random' test data
    theta1 = diw.randInitializeWeights(input_layer_size, hidden_layer_size)
    theta2 = diw.randInitializeWeights(hidden_layer_size, num_labels)

    # Reusing debugInitializeWeights to genete X
    X = diw.randInitializeWeights(input_layer_size - 1, m)
    X1 = np.hstack((np.ones((X.shape[0])).reshape(X.shape[0], 1), X))
    y = 1 + np.mod(np.arange(1, m + 1), num_labels)

    # Unroll parameters
    nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

    def cost_func(p):
        return ncf.ex3_nn(X, y, p, num_labels, X1, hidden_layer_size, lmd)


    cost, grad = cost_func(nn_params)
    numgrad = cng.compute_numerial_gradient(cost_func, nn_params)
    print(np.c_[grad, numgrad])

