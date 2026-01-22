import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	
    final_hidden_state = np.array(initial_hidden_state)
    W_x = np.array(Wx).T
    W_h = np.array(Wh).T
    bias = np.array(b).T

    for X_t in input_sequence:
        X = np.array(X_t).T
        final_hidden_state = np.tanh((X @ W_x) + (final_hidden_state @ W_h) + bias)
        
    return np.round(final_hidden_state, decimals=4)

### TESTING

input_sequence = [[1.0], [2.0], [3.0]]
initial_hidden_state = [0.0]
Wx = [[0.5]]  # Input to hidden weights
Wh = [[0.8]]  # Hidden to hidden weights
b = [0.0]     # Bias

print(rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b))