import numpy as np
import random

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    
    test_size = int(X.shape[0] / k)
    indices = np.array([i for i in range(X.shape[0])])
    
    if shuffle == True:
        np.random.shuffle(indices)
    
    indices = indices.tolist()
    retval = []
    
    for i in range(k):
        retval.append((indices[0 : test_size*i] + indices[test_size*(i+1) : X.shape[0]], indices[test_size*i : test_size*(i+1)]))    
    
    return retval

print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=True))
        # X_train = np.delete(X, range(k*i, (k*(i+1))))
        # y_train = np.delete(y, range(k*i, (k*(i+1))))
        
        # X_test = np.delete(X, range(k*i, (k*(i+1))))
        # y_train = np.delete(y, range(k*i, (k*(i+1))))
        
        