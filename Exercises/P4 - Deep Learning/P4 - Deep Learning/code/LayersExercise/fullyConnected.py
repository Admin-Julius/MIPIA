import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size, batch_size, delta):

        # initialize the weights randomly
        self.weights = np.random.rand(output_size, input_size + 1)

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((input_size + 1, batch_size))  # "pre-allocation"

        # allow individual learning rates per hidden layer
        self.delta = delta

    def forward(self, input_tensor):

        # put together the activations from the input_tensor
        # add an additional row of ones to include the bias (such that w^T * x + b becomes w^T * x equivalently)
        # TODO
        input_tensor = np.vstack((input_tensor, np.ones((1, input_tensor.shape[1]))))
        self.activations = input_tensor
        # perform the forward pass just by matrix multiplication
        layer_output = np.dot(self.weights, input_tensor)  # TODO
        return layer_output

    def backward(self, error_tensor):

        # update the layer using the learning rate and E * X^T,
        # where E is the error from higher layers and X are the activations stored from the forward pass
        #
        # 1. calculate the error for the next layers using the transposed weights and the error
        # TODO
        error1n = np.dot(self.weights.T, error_tensor)
        # 2. update this layer's weights
        # TODO
        self.weights = self.weights - np.dot(self.delta*error_tensor, self.activations.T)

        # the bias of this layer does not affect the layers before, so delete it from the return value
        error_tensor_new = np.delete(error1n, -1, 0)  # TODO
        return error_tensor_new
