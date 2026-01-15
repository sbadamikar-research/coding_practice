import numpy as np

def rref(matrix):
    n_rows = matrix.shape[0]
    for row in range(n_rows):
        if np.sum(matrix[row, :] != 0) == 0:
            if row == (n_rows-1):
                return matrix
                
            matrix[[row, row+1]] = matrix[[row+1, row]]

        if (matrix[row, row] == 0):
            if (row == (n_rows-1)):
                return matrix    
            matrix[[row, row+1]] = matrix[[row+1, row]]
            print(matrix)

        matrix[row, :] = matrix[row, :] / matrix[row, row]
        for i in range(matrix.shape[0]):
            if i == row:
                continue

            matrix[i, :] = matrix[i, :] - (matrix[i, row] * matrix[row, :])
            print(matrix)

    return matrix 

############################################
###               TESTING                ###
############################################
matrix = np.array([ [1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22] ])
print(rref(matrix))

matrix = np.array([ [0, 2, -1, -4], [2, 0, -1, -11], [-2, 0, 0, 22] ], dtype=float)
print(rref(matrix))

# matrix = np.array([ [1, 2, -1], [2, 4, -1], [-2, -4, -3]])
# print(rref(matrix))
