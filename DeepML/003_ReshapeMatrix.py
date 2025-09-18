import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:

    matrix = np.array(a)
    try:
        reshaped_matrix = matrix.reshape(new_shape).tolist()
    except ValueError as e:
        reshaped_matrix = []

    return reshaped_matrix