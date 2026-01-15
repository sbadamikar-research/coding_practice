import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):

    # Loss function = MSE = 1/n_samples * sum [error * error]
    # Error = y - prediction
    # Prediction = w * x + constants

    # dL/dw = [dL / de] * [de / dpred] * [dpred / dw]
    # dL/de = 1/n_samples * sum [ 2 * error] = 2/n_samples * sum (error)
    # de / dpred = -1
    # dpred / dw = x

    # dL/dw = -2/n_samples * sum [ x * error ]

    n_samples, _ = X.shape

    for iterations in range(n_iterations):
        if (method == 'batch'):
            error = y.T - (X @ weights.T)
            grad = (-2 / n_samples)  * (error @ X)
            weights = weights - (learning_rate * grad)

        elif (method == 'stochastic'):
            for i, x in enumerate(X):
                error = y[i] - (x @ weights.T)
                grad = -2 * error * x
                weights = weights - (learning_rate * grad)
        
        elif (method == 'mini_batch'):
            batches = np.ceil(n_samples/batch_size).astype(int)
            for b in range(batches + 1):
                start = b * batch_size
                end = -1 if b == batches else (batches * (b+1))
                X_b = X[start:end, :]
                y_b = y[start:end]

                error = y_b.T - (X_b @ weights.T)
                grad = (-2 / batch_size)  * (error @ X_b)
                weights = weights - (learning_rate * grad)
        
    return weights

# # Test Batch Gradient Descent 
# X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
# y = np.array([2, 3, 4, 5]) 
# weights = np.zeros(X.shape[1])
# learning_rate = 0.01
# n_iterations = 100
# print(gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch'))

# # Test Mini-Batch Gradient Descent 
# X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
# y = np.array([2, 3, 4, 5]) 
# weights = np.zeros(X.shape[1]) 
# learning_rate = 0.01 
# n_iterations = 100 
# batch_size = 2 
# print(gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch'))

# Test Stochastic Gradient Descent
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
y = np.array([2, 3, 4, 5]) 
weights = np.zeros(X.shape[1]) 
learning_rate = 0.01 
n_iterations = 100
print(gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic'))