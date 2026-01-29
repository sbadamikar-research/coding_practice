import numpy as np

def matrix_image(matrix: np.ndarray):
	
    A = np.array(matrix, dtype=float)
    col_max = None

    for row in range(A.shape[0]):
        if (A[row, row] == 0.0):
            max_row = row + np.argmax(A[row::, row])
            A[[row, max_row]] = A[[max_row, row]]
        
        # Pivot not found?
        if (A[row, row] == 0.0):
            break
        
        A[row, :] = A[row, :] / A[row, row]

        for r in range(A.shape[0]):
            # Don't zero out the pivot
            if (r == row):
                continue

            scale = A[r, row] / A[row, row]
            A[r, :] -= (scale * A[row, :])
            A[r, :] = np.round(A[r, :], 3)

        col_max = row

    if col_max is None:
        return []
    else:
        return matrix[:, 0:(col_max+1)]


### TESTING

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_image(matrix))