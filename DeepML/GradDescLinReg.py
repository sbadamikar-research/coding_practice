import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))
    
    for _ in range(iterations):
        y_pred = X.dot(theta)
        diff = y - y_pred
        
        gradient = (1 / m) * X.T.dot(diff)
        
        delta = alpha * gradient
        theta += delta
    
    return theta

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
alpha = 0.01
iterations = 1000
print(linear_regression_gradient_descent(X, y, alpha=alpha, iterations=iterations))