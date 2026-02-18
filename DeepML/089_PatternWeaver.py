import numpy as np

def softmax(values):
	exp_values = np.exp(values)
	return exp_values / exp_values.sum(axis=1, keepdims=True)

def pattern_weaver(n, crystal_values, dimension):
	
	values = np.reshape(np.array(crystal_values), 
					 shape=(n, 1))

	scores = (values @ values.T) / (np.sqrt(dimension))
	probablistic_scores = softmax(scores)

	x = probablistic_scores @ values

	return np.round(x, 3)


print(pattern_weaver(5, [4, 2, 7, 1, 9], 1))