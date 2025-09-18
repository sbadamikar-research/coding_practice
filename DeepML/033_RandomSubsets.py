import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    
    retval = []
    
    y = np.reshape(y, [-1, 1])
    data = np.concatenate([X, y], axis=1)
    
    np.random.seed(seed)
    for _ in range(n_subsets):
        idx = range(len(X))
        if (replacements):
            subset_size = int(X.shape[0])
        else:
            subset_size = int(X.shape[0]/2)
        subset = np.random.choice(idx, subset_size, replace=replacements)
        subsetX = [X[i] for i in subset]
        subsetY = [y[i] for i in subset]
        retval.append([subsetX, subsetY])
        
    return retval
    
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]) 
y = np.array([1, 2, 3, 4, 5]) 
print(get_random_subsets(X,y, 3, False, seed=42))