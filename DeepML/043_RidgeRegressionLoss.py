import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	
    prediction = X @ w.T
    error = y_true.T - prediction

    mean_square_error = (error.T @ error) / X.shape[0]
    penalty = alpha * (w @ w.T)

    return (mean_square_error + penalty)


############################################
###               TESTING                ###
############################################

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
w = np.array([0.2, 2])
y_true = np.array([2, 3, 4, 5])
alpha = 0.1

loss = ridge_loss(X, w, y_true, alpha)
print(loss)