import numpy as np

def rref(matrix: np.ndarray):
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]

    for r in range(n):

        if matrix[r::, r].max() == 0:
            return matrix
        
        # Swap to get the largest pivot of remaining rows
        pivot_r = matrix[r::, r].argmax() + r
        matrix[[r, pivot_r]] = matrix[[pivot_r, r]]

        matrix[r, :] = matrix[r, :] / matrix[r, r] 

        # Pivot about 
        factors = (matrix[:, r].copy()).reshape(-1, 1)
        factors[r, -1] = 0
        row = matrix[r, :].reshape(1, -1)
        matrix = matrix - (factors @ row)

    return matrix

############################################
###               TESTING                ###
############################################
# matrix = np.array([ [1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22] ], )
# print(rref(matrix))

# matrix = np.array([ [0, 2, -1, -4], [2, 0, -1, -11], [-2, 0, 0, 22] ], dtype=float)
# print(rref(matrix))

matrix = np.array([ [1, 2, -1], [2, 4, -1], [-2, -4, -3]])
print(rref(matrix))
