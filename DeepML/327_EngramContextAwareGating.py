# Implement the context-aware gating mechanism from the Engram architecture (DeepSeek).
# This mechanism dynamically modulates retrieved static N-gram embeddings based on the current hidden state context.

# The Engram module retrieves static embeddings from an N-gram memory table, but these embeddings are 
# context-independent and may contain noise from hash collisions or polysemy.
# The context-aware gating mechanism resolves this by using the hidden state (which has aggregated global context) 
# to compute a scalar gate that suppresses irrelevant memory.

# Given:
#     h: Hidden states of shape (T, d) representing contextualized token representations
#     e: Retrieved memory embeddings of shape (T, d_mem) from the N-gram lookup
#     W_K: Key projection matrix of shape (d_mem, d)
#     W_V: Value projection matrix of shape (d_mem, d)

# Your function should return the gated output tensor of shape (T, d).

import numpy as np


def engram_context_gating(h: np.ndarray, e: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Implement Engram context-aware gating mechanism.

    Args:
        h: Hidden states of shape (T, d)
        e: Retrieved memory embeddings of shape (T, d_mem)
        W_K: Key projection matrix of shape (d_mem, d)
        W_V: Value projection matrix of shape (d_mem, d)
        eps: Small constant for numerical stability in RMSNorm

    Returns:
        Gated output of shape (T, d)
    """
    
    k = e @ W_K         # Creates list of T keys with d features
    v = e @ W_V         # Creates list of T values with d features

    h_sq_means = np.mean( (h * h), axis=1, keepdims=True)
    rms = np.sqrt(h_sq_means + eps)
    h_norm = h / rms

    k_sq_means = np.mean( (k * k), axis=1, keepdims=True)
    rms = np.sqrt(k_sq_means + eps)
    k_norm = k / rms

    h_dot_k = np.sum(h_norm * k_norm, axis=1, keepdims=True)

    alpha = 1 / ( 1 + np.exp(-1 * h_dot_k / np.sqrt(h.shape[1])))

    return alpha * v


h = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]) 
e = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7]]) 
W_K = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]) 
W_V = np.array([[0.5, 0.4], [0.4, 0.3], [0.3, 0.2], [0.2, 0.1]])
result = engram_context_gating(h, e, W_K, W_V) 
print(result.shape)