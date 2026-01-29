def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    
    vals = []
    row_idx = []
    col_ptr = [0]

    count = 0
    for col in range(len(dense_matrix[0])):
        for row in range(len(dense_matrix)):
            val = dense_matrix[row][col]
            if (val != 0):
                vals.append(val)
                row_idx.append(row)
                count += 1
            
        col_ptr.append(count)
    
    return vals, row_idx, col_ptr
