import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
    """

    n_samples, n_features = X.shape
    X = np.concatenate((np.ones(shape=(n_samples, 1)), X), axis=1)
    n_classes = y.max() + 1
    params = np.zeros(shape=(n_features + 1, n_classes))

    indicator_fn = np.zeros(shape=(n_samples, n_classes))
    losses = []

    for i in range(n_samples):
        indicator_fn[i, y[i]] = 1
    
    for _ in range(iterations):
        P = np.exp(X @ params)
        P = P / P.sum(axis=1, keepdims=True)

        # http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
        loss = -1 * (indicator_fn * np.log(P)).sum()
        losses.append(np.round(loss, 4))
        grad = -1 * X.T @ (indicator_fn - P)

        params = params - (learning_rate * grad)

    return np.round(params.T, 4), losses

############################################
###               TESTING                ###
############################################

print(train_softmaxreg(np.array([[0.5, -1.2], [-0.3, 1.1], [0.8, -0.6]]), np.array([0, 2, 1]), 0.01, 10))
