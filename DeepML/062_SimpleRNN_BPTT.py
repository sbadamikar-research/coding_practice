import numpy as np
class SimpleRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.W_hy = np.random.randn(output_size, hidden_size)*0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        self.predictions = []
        self.hidden_states = []

    
    def forward(self, X: np.ndarray):
        """
        Forward pass through the RNN for a given sequence of inputs.
        """

        self.predictions = []
        hidden_state = np.zeros(shape=(self.hidden_size, 1))
        self.hidden_states = [hidden_state]
        for x in X:
            x_in = np.reshape(x, (1, -1))
            weighted_input = self.W_xh @ x_in.T
            weighted_hidden = self.W_hh @ hidden_state
            hidden_state = np.tanh( weighted_input + weighted_hidden + self.b_h)
            pred = (self.W_hy @ hidden_state) + self.b_y
            
            self.hidden_states.append(hidden_state)
            self.predictions.append(pred.T)

        return self.predictions

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        T = len(self.predictions) - 1

        # for timestep i
        # L[i] = 0.5/T * sum{i->n}(error[i]**2)
        # dL/derror[i] = 0.5/T * 2 * sum{i->n}(error[i]) = sum{i->n}(error[i])
        
        # error[i] = pred[i] - y[i]
        # derror/dpred[i] = 1
        # dL/dpred[i] = dL/derror[i] * derror/dpred[i]
        # dL/dpred[i] = sum{i->n}(error[i])
        
        # pred[i] = W_hy * h[i] + b_y
        # dpred/dWhy[i] = h[i]

        # dL/dWhy[i] = dL/dpred[i] * dpred/dWhy[i] 
        # dL/dWhy[i] = sum{i->n}(error[i] * h[i])                           <---- Gradient W_hy

        # dpred/dby[i] = 1
        # dL/dby = dL/dpred[i] * dpred_dby[i]
        # dL/dby = sum{i->n}(error[i])                                      <----- Gradient b_y

        # dpred/dh[i] = W_hy 
        # dL/dh[i] = dL/pred[i] * dpred/dh[i]
        # dL/dh[i] = sum{i->n}(error[i] * W_hy)
        # dL/dh[i] = (error[i] * W_hy) + sum{i+1->n}(error[i] * W_hy)
        # dL/dh[i] = (error[i] * W_hy) + dL/dh[i+1]                         <----- Optimization step for computation. 
        #          = (error[i] * W_hy) + dL/dh[next]                        Note that the derivate is for "current" hidden state 
        #                                                                   but at a later time step.
        
        # h[i] = tanh((W_xh * X[i]) + (W_hh * h[i-1]) + (b_h))
        # Let,
        # h_raw[i] = (W_xh * X[i]) + (W_hh * h[i-1]) + (b_h)

        # h[i] = tanh(a[i])
        # dh/dh_raw[i] = 1 - (tanh(a[i]) ** 2)
        # dh/dh_raw[i] = 1 - (h[i]**2)
        # dL/dh_raw[i] = dL/dh[i] * dh/dh_raw[i]

        # a = (W_xh * x) + (W_hh * h[i-1]) + (b_h)
        # dh_raw/dWhh[i] = h[i-1]
        # dh_raw/dWxh[i] = X
        # dh_raw/dbh[i] = 1

        # h_prev = h[i-1]
        # dh_raw/dh_prev[i] = W_hh

        # dL/dWhh[i] = sum{i->n}(dL/dh_raw[i] * dh_raw/dWhh[i])             <---- Gradient W_hh
        # dL/dWxh[i] = sum{i->n}(dL/dh_raw[i] * dh_raw/dWxh[i])             <---- Gradient W_xh
        # dL/dbh[i] = sum{i->n}(dL/dh_raw[i] * dh_raw/dbh[i])               <---- Gradient b_h
        
        # At a previous step j, 
        # dL/dh[j+1] = dL/h_prev[i] = dL/dh_raw * dh_raw/dh_prev            <---- dL/dh[next]

        # https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2

        dh_next = np.zeros_like(self.hidden_states[0])

        dL_dWhy = np.zeros_like(self.W_hy)
        dL_dWhh = np.zeros_like(self.W_hh)
        dL_dWxh = np.zeros_like(self.W_xh)

        dL_dby = np.zeros_like(self.b_y)
        dL_dbh = np.zeros_like(self.b_h)

        for i in range(T, -1, -1):
            y_true = y[i]
            y_pred = self.predictions[i]
            h = self.hidden_states[i+1]
            x = np.reshape(X[i], (1, -1))

            error = np.reshape((y_pred - y_true), (-1, 1))

            dL_dWhy += error @ h.T
            dL_dby += error

            dL_dh = (self.W_hy.T @ error ) + dh_next
            dL_dh_raw = dL_dh * (1 - (h * h))

            dL_dWhh += dL_dh_raw @ self.hidden_states[i].T
            dL_dWxh += dL_dh_raw @ x
            dL_dbh += dL_dh_raw

            dh_next = self.W_hh.T @ dL_dh_raw

        self.W_hy -= (learning_rate * dL_dWhy)
        self.W_hh -= (learning_rate * dL_dWhh)
        self.W_xh -= (learning_rate * dL_dWxh)
        self.b_y -= (learning_rate * dL_dby)
        self.b_h -= (learning_rate * dL_dbh)

### TESTING

np.random.seed(42)
input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])

# Initialize RNN
rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)

for epoch in range(100):
    # Forward pass
    output = rnn.forward(input_sequence)

    # Backward pass
    rnn.backward(input_sequence, expected_output, learning_rate=0.01)

print(output)
# The output should show the RNN predictions for each step of the input sequence