import numpy as np
from collections import Counter

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    data = np.array(data)

    if (strategy == 'mean'):
        replace_with = np.nanmean(data, axis=0)
    elif (strategy == 'median'):
        replace_with = np.nanmedian(data, axis=0)
    elif (strategy == 'mode'):
        replace_with = []
        for col in range(data.shape[1]):
            counts = Counter(data[:, col])
            replace_with.append(counts.most_common(1)[0][0])
    else:
        return data
    
    return np.where(np.isnan(data), replace_with, data)

### TESTING

data = [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]
strategy = 'mode'

print(impute_missing_data(data, strategy))