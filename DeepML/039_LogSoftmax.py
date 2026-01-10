import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	
    scores = np.array(scores)
    shifted_scores = scores - max(scores)
    log_smax = shifted_scores - np.log((np.exp(shifted_scores)).sum())
    return log_smax

print(log_softmax(np.array([1, 2, 3])))