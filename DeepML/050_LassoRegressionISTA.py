import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    w = w * (np.abs(w) > threshold)
    return w


def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    # L = 0.5/n_samples * sum(errors^2)
    # dL/derror = 1/n_samples * sum(errors)
    
    # error = pred - y
    # derror/dpred = 1

    # pred = w * x + b
    # dpred/dw = x
    # dpred/db = 1

    ## w_grad = dL/dw = 1/n_samples * sum(errors * 1 * x) = 1/n_samples * sum (x * errors) 
    ## b_grad = dL/db = 1/n_samples * sum(errors * 1 * 1) = 1/n_samples * sum (errors)

    i = 0

    while (i < max_iter):
        pred = (X @ weights.T) + bias
        errors = pred - y.T

        w_grad = (X.T @ errors) / n_samples
        b_grad = np.sum(errors) / n_samples
        
        # Gradient descent on MSE Loss
        weights = weights - (learning_rate * w_grad)
        bias = bias - (learning_rate * b_grad)

        # Soft thresholding
        weights = soft_threshold(weights, (learning_rate * alpha))

        i = i + 1
    
    return (weights, bias)



############################################
###               TESTING                ###
############################################


X = np.array([[1, 0.01], [2, 0.02], [3, 0.03], [4, 0.04], [5, 0.05]])
y = np.array([2, 4, 6, 8, 10])
weights, bias = l1_regularization_gradient_descent(X, y, alpha=0.5, learning_rate=0.01, max_iter=1000)

print(weights, bias)