import numpy as np

def divide_on_feature(X, feature_i, threshold):
    passing_data = []
    failed_data = []

    for data in X:
        if (data[feature_i] < threshold):
            passing_data.append(data)
        else:
            failed_data.append(data)
    
    return [failed_data, passing_data]
