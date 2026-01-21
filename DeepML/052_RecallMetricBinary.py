# How well can the model classify positive results?

import numpy as np

def recall(y_true: np.ndarray, y_pred: np.ndarray):
    if not y_true.sum():
        return 0.0

    return np.round( ((y_pred * y_true).sum() / y_true.sum()), 3 )

### TESTING

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))