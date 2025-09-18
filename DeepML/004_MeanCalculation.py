import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    dict = {
        'row': 1,
        'column': 0
    }
    
    means = np.mean(np.array(matrix), axis=dict[mode]).tolist()
    return means