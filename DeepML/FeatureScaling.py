import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here

    means = data.mean(axis=0)
    std_dev = data.std(axis=0)
    minval = data.min(axis=0)
    maxval = data.max(axis=0)
    rangeval = maxval - minval
    
    print(means, std_dev)
    
    standardized_data = []
    normalized_data = []
    for row in data:    
        standardized_data.append((row - means) / std_dev)
        normalized_data.append((row - minval) / rangeval)
    
    return standardized_data, normalized_data

data = np.array([[1, 2], [3, 4], [5, 6]])
print(feature_scaling(data))
