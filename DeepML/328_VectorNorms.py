import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    if (norm_type == 'l1'):
        return np.sum(np.abs(arr)).astype(float)
    elif (norm_type == 'l2' or norm_type == 'frobenius'):
        return np.sqrt(np.sum(arr * arr)).astype(float)
    
    return 0


arr = np.array([1, -2, 3])
norm_type = 'l1'
print(compute_norm(arr, norm_type))