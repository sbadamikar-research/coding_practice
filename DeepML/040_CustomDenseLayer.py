import numpy as np

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS

class Layer(object):

    def set_input_shape(self, shape):

        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
        self.optimizer = None
        self.learning_rate = 0.01
        self.batch_size = 0

    def initialize(self, optimizer: any = None):

        if optimizer is not None:
            self.optimizer = optimizer

        # Every row represents the weight of each feature
        # Every column represents a neuron in the hidden layer.
        self.W = np.random.uniform(size=(self.input_shape[0], self.n_units),
                                   low=(-1 / np.sqrt(self.input_shape[0])),
                                   high=(1 / np.sqrt(self.input_shape[0])))
        
        # Every row represents the bias of the corresponding hidden neuron
        self.w0 = np.zeros(shape=(1, self.n_units))

    def forward_pass(self, X: np.ndarray):
        self.layer_input = X
        self.batch_size = self.layer_input.shape[0]

        output = self.layer_input @ self.W
        output = output + (np.full(shape=(self.batch_size, 1), fill_value=1) @ self.w0)

        return output

    def backward_pass(self, accum_grad):
        accum_grad = np.array(accum_grad)

        grad_W = self.layer_input.T @ accum_grad
        grad_w0 = np.sum(accum_grad, axis=0)
        grad_X = accum_grad @ self.W.T

        if self.optimizer is None:
            self.W = self.W - (self.learning_rate * grad_W)
            self.w0 = self.w0 - (self.learning_rate * grad_w0)
        else:
            self.W = self.optimizer.update(self.W, grad_W)
            self.w0 = self.optimizer.update(self.w0, grad_w0)
        
        return grad_X

    def parameters(self):
        return np.size(self.W) + np.size(self.w0)

    def output_shape(self):
        return (self.n_units,)

############################################
###               TESTING                ###
############################################
# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))

# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)

