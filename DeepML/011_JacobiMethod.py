import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

    x = np.zeros_like(b, dtype=float)
    
    for _ in range(n):
        
        temp_x = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            for j in range (len(x)):
                if (j != i):
                    temp_x[i] += A[i][j] * x[j]
                
            temp_x[i] = (b[i] - temp_x[i]) / A[i][i]
        
        x = temp_x.copy()
        
    return x.tolist()