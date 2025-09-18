import numpy as np

def make_diagonal(x):
    retval = np.ndarray([len(x), len(x)], dtype=float)
    for i, val in enumerate(x):
        retval[i][i] = val
    return retval