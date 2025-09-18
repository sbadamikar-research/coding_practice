import numpy as np
import pandas as pd

def calculate_correlation_matrix(X, Y=None):
    dfX = pd.DataFrame(X)
    return dfX.corr().to_numpy()
    
print(calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])))