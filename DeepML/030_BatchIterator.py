import numpy as np
import math

def batch_iterator(X, y=None, batch_size=64):
    retval = []
    idx = 0
    for _ in range(math.floor(len(X) / batch_size)):
        valX = []
        valY = []
        for _2 in range(batch_size):
            if(idx >= len(X)):
                continue
            
            valX.append(X[idx])
            if (y is not None):
                valY.append(y[idx])
            
            idx += 1
        
        if (len(valY) > 0):
            retval.append([valX, valY])
        else:
            retval.append(valX)
    
    return retval