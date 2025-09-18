import math
import numpy as np

def sigmoid(vals: np.ndarray) -> float:
	
    for i, val in enumerate(vals):
        vals[i] = round((1 / (1 + math.exp(-1 * val))), 4)
    
    return vals

def train_neuron(features: np.ndarray, 
                 labels: np.ndarray, 
                 initial_weights: np.ndarray, 
                 initial_bias: float, 
                 learning_rate: float, 
                 epochs: int) -> (np.ndarray, float, list[float]):
    
    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []
    
    n = features.shape[0]
    for _ in range(epochs):
        prediction = sigmoid((features @ updated_weights) + updated_bias)
        
        errors = labels - prediction 
        mse_values.append((errors.T @ errors)/n)
        
        dL_dz = ((-2 / n) * errors) * (prediction * (1 - prediction))
        
        bias_gradient = np.sum(dL_dz)
        updated_bias = updated_bias - (learning_rate * bias_gradient)

        weight_gradient = features.T @ dL_dz
        updated_weights = updated_weights - (learning_rate * weight_gradient)
        
    return np.round(updated_weights, 4), np.round(updated_bias, 4), np.round(mse_values, 4)

features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
labels = [1, 0, 0]
initial_weights = np.array([0.1, -0.2])
initial_bias = 0.0
learning_rate = 0.1
epochs = 2

print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))