import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	
    # Standardizing the data (mean=0, variance=1)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Covariance matrix
    cov_matrix = ((data - np.mean(data, axis=0)).transpose() @ (data - np.mean(data, axis=0))) / data.shape[1]
    
    # Determine and sort eigen values and vectors
    eigenvals, eigenvectors = np.linalg.eig(cov_matrix)
    unsorted_eigens = zip(eigenvals, eigenvectors.transpose())
    eigens = sorted(unsorted_eigens)
    
    if (k > len(eigens)):
        k = len(eigens)
    
    retval = []
    for i in range(1, k+1):
        retval.append(np.round(eigens[(-1*i)][1], 4))
    
    retval = np.array(retval).transpose().tolist()
    
    return retval

data = np.array([[1, 2], [3, 4], [5, 6]])
k = 1

print(pca(data, k))

data = np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]])
k = 2
print(pca(data, k))
# [[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]