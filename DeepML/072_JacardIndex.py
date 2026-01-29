import numpy as np

def jaccard_index(y_true, y_pred):
	
    print(y_true == y_pred)
    intersect = (y_pred * (y_true == y_pred)).sum()
    union = ((y_true + y_pred) > 0).sum()

    if union == 0:
        return 1
    
    result = intersect/union
    return round(result, 3)

### TESTING

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
print(jaccard_index(y_true, y_pred))