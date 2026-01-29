import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    return round( (1 - (probabilities @ probabilities.T)), 3)

### TESTING

y = [0, 1, 1, 1, 0]
print(gini_impurity(y))