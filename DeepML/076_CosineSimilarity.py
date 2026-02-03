import numpy as np

def cosine_similarity(v1, v2):
	
    v1v2_dot = v1 @ v2.T

    v1_mag = np.sqrt(v1 @ v1.T)
    v2_mag = np.sqrt(v2 @ v2.T)

    return (v1v2_dot / (v1_mag * v2_mag))


### TESTING

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(round(cosine_similarity(v1, v2), 3))