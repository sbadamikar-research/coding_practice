import numpy as np
def precision(y_true, y_pred):
	
    if not y_pred.sum():
        return 1

    true_positives = y_pred @ (y_pred * y_true).T
    false_positives = y_pred @ (y_pred * (1 - y_true)).T

    return (true_positives / (true_positives + false_positives))


############################################
###               TESTING                ###
############################################

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

result = precision(y_true, y_pred)
print(result)