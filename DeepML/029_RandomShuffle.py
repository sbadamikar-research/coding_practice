import numpy as np

def shuffle_data(X, y, seed=None):
	idx = range(0, X.shape[0])
	
	y = np.reshape(y, [-1,1])
	data = np.concatenate([X, y], axis=1)
	
	np.random.seed(seed)
	np.random.shuffle(data)
	y = data[:, -1]
	X = np.delete(data, -1, axis=1)
 
	return X, y
	
print(shuffle_data(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4]), seed=42))
# (array([[3, 4], [7, 8], [1, 2], [5, 6]]), array([2, 4, 1, 3]))

print(shuffle_data(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), np.array([10, 20, 30, 40]), seed=24))
# (array([[4, 4],[2, 2],[1, 1],[3, 3]]), array([40, 20, 10, 30]))