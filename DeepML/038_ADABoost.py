# Write a Python function adaboost_fit that implements the fit method for an AdaBoost classifier. 
# The function should take in a 2D numpy array X of shape (n_samples, n_features) representing the dataset, 
# a 1D numpy array y of shape (n_samples,) representing the labels, and an integer n_clf representing the number of classifiers. 
# The function should initialize sample weights, find the best thresholds for each feature, calculate the error, update weights, 
# and return a list of classifiers with their parameters.

import numpy as np
import math

def fit_clf(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    parameters = {'polarity': 1, 'threshold': 0, 'feature_index': 0, 'alpha': 0}

    feature_thresholds = np.zeros(X.shape[1])
    feature_accuracies = np.zeros(X.shape[1])

    for idx_feature in range(X.shape[1]):
    
        for t in np.unique(X[:, idx_feature]):
            threshold_accuracy = 0

            for i, val in enumerate(X[:, idx_feature]):
                p = -1 if val < t else 1
                threshold_accuracy += w[i] * (p == y[i])
            
            # Use the higher threshold accuracy as polarity might be flipped.
            threshold_accuracy = max(threshold_accuracy, (1-threshold_accuracy))

            if (threshold_accuracy > feature_accuracies[idx_feature]):
                # print(f'{idx_feature}:{t} - {threshold_accuracy}')
                feature_accuracies[idx_feature] = threshold_accuracy
                feature_thresholds[idx_feature] = t

    parameters['feature_index'] = int(feature_accuracies.argmax())
    parameters['threshold'] = float(feature_thresholds[parameters['feature_index']])

    return parameters


def adaboost_fit(X: np.ndarray, y: np.ndarray, n_clf: int):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))

    clfs = []

    for c in range(n_clf):
        clf = fit_clf(X, y, w)
        print(clf)
        error = 0
        p = np.zeros(y.shape)
        for i, sample in enumerate(X):
            p[i] = -1 if sample[clf['feature_index']] < clf['threshold'] else 1
            error = error + (w[i] * int(p[i] != y[i]))

        if error > 0.5:
            error = 1 - error
            clf['polarity'] = -1 * clf['polarity']
            p = -1 * p
        
        
        clf['alpha'] = 0.5 * math.log( (1 - error) / (error + 1e-10))
        
        w_sum = 0
        for i in range(n_samples):
            w[i] = w[i] * math.exp(-1 * clf['alpha'] * y[i] * p[i] )
            w_sum = w_sum + w[i]    
        w = w / w_sum
        
        clfs.append(clf)

    return clfs


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])
n_clf = 3
print(adaboost_fit(X, y, n_clf))

X = np.array([[8, 7], [3, 4], [5, 9], [4, 0], [1, 0], [0, 7], [3, 8], [4, 2], [6, 8], [0, 2]]) 
y = np.array([1, -1, 1, -1, 1, -1, -1, -1, 1, 1]) 
n_clf = 2 
print(adaboost_fit(X, y, n_clf))