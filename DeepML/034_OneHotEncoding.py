import numpy as np
import pandas as pd

def to_categorical(x, n_col=None):
    
    df = pd.DataFrame(x)
    unique_values = pd.unique(df[0])
    unique_values.sort()
    
    if (n_col is None):
        n_empty = 0
    else:
        n_empty = n_col - len(unique_values)
    
    for i in range(0, n_empty):
        df[i+1] = 0
    
    for i, val in enumerate(unique_values):
        val_map = {val: 1}
        idx = i + n_empty + 1
        df[idx] = df[0].map(val_map)
        df[idx] = df[idx].fillna(0)
    
    df.pop(0)
    return df.to_numpy()
    
print(to_categorical(np.array([0, 1, 2, 1, 0])))

print(to_categorical(np.array([3, 1, 2, 1, 3]), 4))