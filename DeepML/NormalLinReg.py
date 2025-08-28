import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    
    Xn = np.array(X)
    y_t = np.array(y)
    
    theta = y_t @ Xn @ np.linalg.pinv(Xn.transpose() @ X)
    
    return theta