def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    
    retval = [[0 for i in range(len(a))] for j in range(len(a[0]))]
    
    for i in range(len(a)):
        for j in range(len(a[i])):
            retval[j][i] = a[i][j]
            
    return retval

print(transpose_matrix([[1,2],[3,4],[5,6]]))