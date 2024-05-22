import numpy as np


def one_hot(x, n):
    return np.eye(n)[x]
