import numpy as np

def f_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    
    recall = (y_pred * y_true).sum() / y_true.sum()
    precision = (y_pred * y_true).sum() / y_pred.sum()

    if (precision + recall) == 0:
        return 0.0
    
    f_score = (1 + beta**2) * ( (precision * recall) / ( (beta**2 * precision) + recall))
    return np.round(f_score, 3)


### TESTING

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
beta = 1

print(f_score(y_true, y_pred, beta))