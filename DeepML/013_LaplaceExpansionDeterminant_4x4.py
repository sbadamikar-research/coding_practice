import numpy as np
def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
    
    if len(matrix) == 1:
        print(matrix[0][0])
        return matrix[0][0]
    
    cofactor = -1
    det = 0
    for i in range(len(matrix)):
        cofactor *= -1
        
        minor = []
        for j in range(len(matrix)):
            if i!=j:
                row = [x for k, x in enumerate(matrix[j]) if k!=0 ]
                minor.append(row)
        
        print(minor)
        det += cofactor * matrix[i][0] * determinant_4x4(minor)
        print([i, det])
    
    return det

a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
print(a)
print(determinant_4x4(a))