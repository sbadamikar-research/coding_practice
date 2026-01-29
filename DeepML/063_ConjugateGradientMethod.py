import numpy as np

def conjugate_gradient(A: np.ndarray, b: np.ndarray, n: int, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """ 
    b = np.reshape(b, (-1, 1))

    if x0 == None:
        x = np.zeros_like(b)
    else:
        x = x0
    
    r = b - (A @ x)
    p = r

    iteration = 0
    while (iteration < n):
        # Determine step size
        r_mag = r.T @ r
        step_size = r_mag / (p.T @ A @ p)
        
        # Calculate x & residual for this step
        x = x + (step_size * p)
        r = r - (step_size * (A @ p))
        new_r_mag = r.T @ r

        # If magnitude of residual is in tolerance
        # x is the solution
        if new_r_mag < tol:
            return x.T
        
        # New direction scaling
        beta = new_r_mag / r_mag
        p = r + (beta * p)

        iteration += 1

    return x.T

### TESTING

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
n = 5
print(conjugate_gradient(A, b, n))
# [0.09090909, 0.63636364]

A = np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]]) 
b = np.array([7, 8, 5]) 
n = 1 
print(conjugate_gradient(A, b, n))
# [1.2627451, 1.44313725, 0.90196078]

A = np.array([[6, 2, 1, 1, 0], [2, 5, 2, 1, 1], [1, 2, 6, 1, 2], [1, 1, 1, 7, 1], [0, 1, 2, 1, 8]]) 
b = np.array([1, 2, 3, 4, 5]) 
n = 100 
print(conjugate_gradient(A, b, n))
# [0.01666667, 0.11666667, 0.21666667, 0.45, 0.5]