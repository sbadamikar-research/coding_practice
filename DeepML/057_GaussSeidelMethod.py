import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    
    if x_ini is None:
        x = np.zeros(shape=(A.shape[1], 1))
    else:
        x = x_ini

    for _ in range(n):
        for i in range(x.shape[0]):
            # Skip x_i
            x[i] = 0           
            x[i] = (b[i] - (A[i, :] @ x)) / A[i, i]

    return x


### TESTING

A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)

# A = np.array([[3, 1], [1, 2]], dtype=float)
# b = np.array([5, 5], dtype=float)

n = 10
print(gauss_seidel(A, b, n))