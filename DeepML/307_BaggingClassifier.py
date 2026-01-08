import numpy as np

class DecisionStump:
    feature = 0
    threshold = 0

    def __init__(self):
        self.feature = 0
        self.threshold = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        max_threshold = np.ceil(X_train.max()).astype(int)

        threshold_accuracy = np.zeros(max_threshold * 10)
        threshold_feature = np.zeros(max_threshold * 10)
        
        for t_i, t in enumerate(np.arange(0, max_threshold, 0.1)):

            feature_accuracy = np.zeros(X_train.shape[1])
            for j in range(X_train.shape[1]):
                
                for i, val in enumerate(X_train[:,j]):
                    p = 0 if val < t else 1
                    if (p == y_train[i]):
                        feature_accuracy[j] = feature_accuracy[j] + 1

                if (feature_accuracy[j] > threshold_accuracy[t_i]):
                    threshold_accuracy[t_i] = feature_accuracy[j]
                    threshold_feature[t_i] = j
        
        threshold_idx = threshold_accuracy.argmax()
        self.threshold = threshold_idx / 10
        self.feature = threshold_feature[threshold_idx].astype(int)

        return self

    def predict(self, X_test: np.ndarray):
        y_test = np.zeros(X_test.shape[0])
        for i, sample in enumerate(X_test):
            y_test[i] = 0 if (sample[self.feature] < self.threshold) else 1
        
        return y_test


def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    rng = np.random.default_rng(seed=seed)
    y_train = np.reshape(y_train, (-1,1))
    data = np.concatenate((X_train, y_train), axis=1)
    stumps = []

    for _ in range(n_estimators):    
        bootstrap_sample = rng.choice(data, size=data.shape[0], replace=True)
        
        stump = DecisionStump()
        stump = stump.fit(bootstrap_sample[:,0:-1], bootstrap_sample[:, -1])
        stumps.append(stump)

    predictions = np.zeros((len(X_test), 1))
    for stump in stumps:
        y_test_i = stump.predict(X_test)
        y_test_i = np.reshape(y_test_i, (-1,1))
        predictions = predictions + y_test_i

    predictions = np.round((predictions / len(stumps))).astype(int)
    return predictions.flatten()

X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 4], [5, 2], [6, 3]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[2, 2], [5, 3]])
n_estimators=5
seed=42
print(bagging_classifier(X_train, y_train, X_test, n_estimators, seed))

X_train = np.array([[0.5, 1.0], [1.0, 0.5], [1.5, 1.5], [2.0, 0.8], [2.5, 2.0], [3.0, 1.2], [3.5, 2.5], [4.0, 1.8]]) 
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1]) 
X_test = np.array([[1.2, 1.0], [2.8, 1.5], [4.5, 2.0]]) 
print(bagging_classifier(X_train, y_train, X_test, n_estimators=10, seed=0))