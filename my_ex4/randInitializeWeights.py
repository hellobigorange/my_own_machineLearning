import numpy as np
import math

def randInitializeWeights(L_in, L_out):
    epsilon_init = math.sqrt(6)/math.sqrt(L_in+L_out)
    theta_init = np.random.random ((L_out, L_in+1))*(2*epsilon_init)-epsilon_init
    return theta_init
