import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    retval = []
    for data in X:
        polynomial_features = [1.] + data.tolist()
        for deg in range(2, degree + 1):
            for val_set in combinations_with_replacement(data, deg):
                element = 1
                for val in val_set:
                    element *= val
                polynomial_features.append(element)
                
        polynomial_features.sort()
        retval.append(polynomial_features)
    
    return retval
                

X = np.array([[2, 3],
              [3, 4],
              [5, 6]])
degree = 2
output = polynomial_features(X, degree)
print(output)