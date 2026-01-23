import numpy as np

def sigmoid(X: np.ndarray):
	
    return (1 / (1 + np.exp(-1 * X)))

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x: np.ndarray, initial_hidden_state: np.ndarray, initial_cell_state: np.ndarray):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """

        hidden_state = initial_hidden_state
        cell_state = initial_cell_state

        for val in x:
            x_t = val.reshape(1, -1)
            state = np.concatenate((hidden_state.T, x_t), axis=1)

            forget_gate = sigmoid( (self.Wf @ state.T) + self.bf )

            input_gate = sigmoid( (self.Wi @ state.T) + self.bi )
            candidates = np.tanh( (self.Wc @ state.T) + self.bc )

            cell_state = (forget_gate * cell_state) + (input_gate * candidates)
            
            output = sigmoid( (self.Wo @ state.T) + self.bo)

            hidden_state = (output * np.tanh(cell_state))
        
        return output, hidden_state, cell_state

### TESTING

# input_sequence = np.array([[1.0], [2.0], [3.0]])
# initial_hidden_state = np.zeros((1, 1))
# initial_cell_state = np.zeros((1, 1))

# lstm = LSTM(input_size=1, hidden_size=1)
# outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

# print(final_h)

input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]]) 
initial_hidden_state = np.zeros((2, 1)) 
initial_cell_state = np.zeros((2, 1)) 

lstm = LSTM(input_size=2, hidden_size=2) # Set weights and biases for reproducibility 
lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 

lstm.bf = np.array([[0.1], [0.2]]) 
lstm.bi = np.array([[0.1], [0.2]]) 
lstm.bc = np.array([[0.1], [0.2]]) 
lstm.bo = np.array([[0.1], [0.2]]) 

outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state) 
print(final_h)