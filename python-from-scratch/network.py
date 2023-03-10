import math
import numpy

class Network:
    """
    custom neural network class with functions for calculating outputs
    using standard feedforward propagation
    """

    def __init__(self, layer_sizes: list[int]):
        self.__layer_count = len(layer_sizes)
        self.__layer_sizes = layer_sizes
        self.__initialize_weights_and_biases()

    def __initialize_weights_and_biases(self):
        '''
        initialize the weights with he initialization
        and biases as zero
        weights are stored in a list of 2D numpy arrays
        biases are stored in a list of 1D numpy arrays / nD vectors
        '''
        self.__weights: list[numpy.ndarray] = []
        for i, previous_layer_size in enumerate(self.__layer_sizes[:-1]):
            next_layer_size = self.__layer_sizes[i + 1]

            standard_deviation = math.sqrt(2.0 / previous_layer_size)

            self.__weights.append(
                numpy.array(
                    [
                        numpy.random.randn(previous_layer_size) * standard_deviation
                        for j in range(next_layer_size)
                    ]
                )
            )

        self.__biases = [
            numpy.array(
                [
                    0.0
                    for j in range(layer_size)
                ]
            )
            for layer_size in self.__layer_sizes[1:]
        ]

    @staticmethod
    def __ReLU(vector: numpy.ndarray) -> numpy.ndarray:
        '''
        the rectified linear activation function
        for a vector
        '''
        return numpy.array([max(0.0, n) for n in vector])

    def calculate_outputs(self, inputs: numpy.ndarray) -> numpy.ndarray:
        '''
        calculate outputs using given inputs
        '''
        current_layer = inputs
        for i in range(self.__layer_count - 1):
            current_layer = self.__ReLU(
                self.__weights[i].dot(current_layer) + self.__biases[i]
            )
        return current_layer
