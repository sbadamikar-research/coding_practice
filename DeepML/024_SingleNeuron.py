import math
import numpy as np

def sigmoid(z: float) -> float:
	#Your code here
	return round((1 / (1 + math.exp(-1 * z))), 4)

def single_neuron_model(features: list[list[float]],
                        labels: list[int], 
                        weights: list[float], 
                        bias: float) -> (list[float], float):
    
    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=float)
    weights = np.array(weights, dtype=float)
    
    probabilities = ((features @ weights) + bias)
    
    mse = 0 
    for i in range(probabilities.shape[0]):
        probabilities[i] = sigmoid(probabilities[i])
        mse += math.pow((probabilities[i] - labels[i]), 2)
    
    mse /= probabilities.shape[0] 
    mse = round(mse, 4)
    
    return probabilities, mse

features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1

print(single_neuron_model(features, labels, weights, bias))