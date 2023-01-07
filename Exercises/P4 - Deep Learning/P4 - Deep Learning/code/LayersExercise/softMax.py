import numpy as np


class SoftMax:

    def __init__(self, categories, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((categories, batch_size))  # "pre-allocation"

    def forward(self, input_tensor):

        # store the activations from the input_tensor
        self.activations = np.copy(input_tensor)

        # apply SoftMax to the scores: e(x_i) / sum(e(x))
        # TODO
        input_tensor = np.exp(input_tensor)
        total = np.sum(input_tensor, axis=0, keepdims=True)
        input_tensor = input_tensor / total
        self.activations = input_tensor

        return self.activations

    def backward(self, label_tensor):

        error_tensor = np.copy(self.activations)

        #  Given:
        #  - the labels are one-hot vectors
        #  - the loss is cross-entropy (as implemented below)
        # Idea:
        # - decrease the output everywhere except at the position where the label is correct
        # - implemented by increasing the output at the position of the correct label
        # Hint:
        # - do not let yourself get confused by the terms 'increase/decrease'
        # - instead consider the effect of the loss and the signs used for the backward pass

        # TODO
        # ...
        for i in range(error_tensor.shape[1]):
            itemindex = np.where(label_tensor[:, i] == 1)
            error_tensor[itemindex, i] -= 1
        return error_tensor

    def loss(self, label_tensor):

        loss = 0

        # iterate over all elements of the batch and sum the loss
        # TODO
        for i in range(self.activations.shape[1]):
            itemindex = np.where(label_tensor[:, 1] == 1)
            loss += -np.log(self.activations[itemindex, i])
        # ... # loss is the negative log of the activation of the correct position

        return loss
