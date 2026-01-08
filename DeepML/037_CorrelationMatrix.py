import numpy as np

def calculate_correlation_matrix(X, Y=None):

    s = X.shape[1]

    if (Y is not None):
        data = np.concatenate((X,Y), axis=1)
    else:
        data = np.concatenate((X,X), axis=1)
    
    corr_mat = np.identity(data.shape[1])

    for i in range(0, data.shape[1]):
        for j in range(i+1, data.shape[1]):
            data_x = data[:,i]
            data_y = data[:,j]

            mean_x = np.mean(data_x)
            mean_y = np.mean(data_y)

            Sxx = 0
            Syy = 0
            Sxy = 0
            for n in range(len(data_x)):
                Sxx = Sxx + ((data_x[n] - mean_x) * (data_x[n] - mean_x))
                Syy = Syy + ((data_y[n] - mean_y) * (data_y[n] - mean_y))
                Sxy = Sxy + ((data_x[n] - mean_x) * (data_y[n] - mean_y))

            corr_mat[i][j] = Sxy / np.sqrt(Sxx * Syy)
            corr_mat[j][i] = corr_mat[i][j]
            
    return corr_mat[0:s, s::]

print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))
    
print(calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])))

print(calculate_correlation_matrix(np.array([[1, 0], [0, 1]]), np.array([[1, 2], [3, 4]])))