

import numpy as np
import math

def sigmoid(z_vector):
    output = []
    for z in z_vector:
        y = 1 / (1 + math.e ** (-1 * z))
        output.append(y)
    return np.array(output)

def jacobian_sigmoid(y_vector):
    derivatives = []
    for output in y_vector:
        d = float(output)*(1 - float(output))
        derivatives.append(d)
    derivatives = np.array(derivatives)
    jacobian = np.diag(derivatives)
    return jacobian
