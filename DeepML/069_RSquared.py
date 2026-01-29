import numpy as np

def r_squared(y_true: np.ndarray, y_pred: np.ndarray):
	
    error = y_true - y_pred
    deviation = y_true - y_true.mean()

    return (1 - ( (error @ error.T) / (deviation @ deviation.T)))


### TESTING

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
print(r_squared(y_true, y_pred))