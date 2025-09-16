import numpy as np

def kernel_fn(data, kernel='linear'):
    retval = np.ndarray([0,0])
    if kernel == 'linear':
        retval = np.dot(data, data.T)
    
    print(retval)
    return retval
    


def pegasos_kernel_svm(data: np.ndarray,
                       labels: np.ndarray,
                       kernel='linear',
                       lambda_val=0.01,
                       iterations=100,
                       sigma=1.0) -> (list, float):

    alphas = np.zeros([data.shape[0], 1], dtype=float)
    b = 0

    for t in range(1, iterations+1):

        # Determine the learning rate for iteration
        learning_rate = 1 / (lambda_val * t)

        # Shrinkage
        alphas = (1 - (learning_rate * lambda_val)) * alphas

        f = (alphas * labels) @ kernel_fn(data, kernel)

    return alphas, b


data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]])
labels = np.array([1, 1, -1, -1])
kernel = 'linear'
lambda_val = 0.01
iterations = 100
sigma = 1.0

alphas, b = pegasos_kernel_svm(
    data, labels, kernel, lambda_val, iterations, sigma)
