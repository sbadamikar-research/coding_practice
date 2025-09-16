import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    
    scores = np.array(scores, dtype=float)
    probabilities = np.zeros_like(scores)
    for i, score in enumerate(scores):
        probabilities[i] = math.exp(score)
    
    probabilities = np.round((probabilities / probabilities.sum()), 4)
    return probabilities

scores = [1, 2, 3]
print(softmax(scores))