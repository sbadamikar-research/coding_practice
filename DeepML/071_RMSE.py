import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
	
    error = y_pred - y_true
    rmse_res = np.sqrt((error * error).sum() / error.size)
    
    return np.round(rmse_res, 3)

### TESTING

# Test Case 2: 2D Array 
y_true2 = np.array([[0.5, 1], [-1, 1], [7, -6]]) 
y_pred2 = np.array([[0, 2], [-1, 2], [8, -5]]) 
print(rmse(y_true2, y_pred2)) 