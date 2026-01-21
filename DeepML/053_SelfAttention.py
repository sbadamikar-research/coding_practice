import numpy as np

def softmax(X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)
    

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    
    attention_score = Q @ K.T / np.sqrt(Q.shape[1])
    attention_weights = softmax(attention_score)
    
    return attention_weights @ V 


### TESTING

Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 2], [3, 4]])

print(softmax(V))

# print(self_attention(Q, K, V))