import numpy as np 
import math

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    
    
    eigenvals, eigenvecs = np.linalg.eig(A.transpose() @ A)
    v2V = dict(zip(eigenvals, np.array(eigenvecs.transpose())))
    v2V = dict(sorted(v2V.items(), reverse=True))
    
    for key in v2V:
        neg_count = [x < 0 for x in v2V[key]].count(True)
        if neg_count == len(v2V[key]):
            v2V[key] = -1 * v2V[key]
        
    S = np.array([math.sqrt(key) for key in v2V])
    V = np.array([v2V[key] for key in v2V])
    
    
    Av = A @ V.transpose()
    Ut = (1/S).transpose() * Av
     
    U = Ut.transpose()
    SVD = (U, S, V)
    
    return SVD

a = [[2, 1], [1, 2]]
b = [[1, 2], [3, 4]]
print(svd_2x2_singular_values(np.array(b)))