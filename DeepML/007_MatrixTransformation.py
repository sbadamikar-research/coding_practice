import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

    npA = np.array(A)
    npT = np.array(T)
    npS = np.array(S)

    if  (np.linalg.det(npT) == 0) or (np.linalg.det(npS) == 0):
        return -1

	try:
        transformed_matrix = (np.linalg.inv(npT) @ npA @ npS).tolist()
    except:
        transformed_matrix = []

    return transformed_matrix