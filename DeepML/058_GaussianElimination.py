import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """

    b = b.reshape(-1, 1)
    M = np.concatenate((A, b), axis=1)
    n = M.shape[0]

    for r in range(n):
        # Swap to get the largest pivot of remaining rows
        pivot_r = M[r::, r].argmax() + r
        M[[r, pivot_r]] = M[[pivot_r, r]]

        # Pivot about r
        next_r = r + 1
        factors = (M[next_r::, r] / M[r, r]).reshape(-1, 1)
        row = M[r, :].reshape(1, -1)

        M[next_r::, :] = M[next_r::, :] - (factors @ row)

    x = []
    for iteration, r in enumerate(range(n-1, -1, -1)):
        sub_val = 0
        for i in range(iteration):
            sub_val += (x[i] * M[r, (n-i-1)])
        
        x.append( (M[r, -1] - sub_val) / M[r, r] )
    
    x.reverse()
    return x

# TESTING


A = np.array([[2, 8, 4], [2, 5, 1], [4, 10, -1]], dtype=float)
b = np.array([2, 5, 1], dtype=float)

print(gaussian_elimination(A, b))
