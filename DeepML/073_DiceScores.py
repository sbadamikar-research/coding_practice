import numpy as np

def dice_score(y_true, y_pred):
	
    intersection = (y_pred * (y_true == y_pred)).sum()
    positive_counts = y_pred.sum() + y_true.sum()

    if positive_counts == 0:
        return 1

    return round(intersection/positive_counts, 3)