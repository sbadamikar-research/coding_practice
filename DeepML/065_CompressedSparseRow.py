import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    
    row_ptr = [0]
    vals = []
    col_idx = []
    count = 0

    for row in dense_matrix:
        for col, val in enumerate(row):
            if (val != 0):
                vals.append(val)
                col_idx.append(col)
                count += 1
        row_ptr.append(count)

    return vals, col_idx, row_ptr


### TESTING

dense_matrix = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 4, 0],
    [1, 0, 0, 5]
]

vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)