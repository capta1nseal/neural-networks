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
        rectified linear activation function
        for a 1D numpy array / nD vector
        '''
        return numpy.array([max(0.0, n) for n in vector])

    def __sigmoid(self, vector: numpy.ndarray) -> numpy.ndarray:
        '''
        sigmoid activation function
        for a 1D numpy array / nD vector
        '''
        return numpy.array([1.0 / (1.0 + pow(math.e, -n)) for n in vector])

    def get_layer_sizes(self) -> list[int]:
        """
        returns the layer sizes of the neural network
        """
        return self.__layer_sizes

    def calculate_output(self, inputs: numpy.ndarray) -> numpy.ndarray:
        '''
        calculate outputs using given inputs
        ReLU activation is used for all layers
        except the final layer, which uses sigmoid activation
        '''
        current_layer = inputs
        for i in range(self.__layer_count - 2):
            current_layer = self.__ReLU(
                self.__weights[i].dot(current_layer) + self.__biases[i]
            )
        return self.__sigmoid(self.__weights[-1].dot(current_layer) + self.__biases[-1])

    def get_state(self, inputs: numpy.ndarray) -> list[numpy.ndarray]:
        """
        get the state of the network including hidden layers
        """
        state = [inputs]
        for i in range(self.__layer_count - 2):
            state.append(
                self.__ReLU(
                    self.__weights[i].dot(state[-1]) + self.__biases[i]
                )
            )
        state.append(self.__sigmoid(self.__weights[-1].dot(state[-1]) + self.__biases[-1]))
        return state

    def get_weights(self, index: int) -> numpy.ndarray:
        """
        get one layer's weights
        """
        return self.__weights[index]

    def get_biases(self, index: int) -> numpy.ndarray:
        """
        get one layer's biases
        """
        return self.__biases[index]

    def adjust_weights(self, weight_adjustments: list[numpy.ndarray]) -> None:
        """
        adjust the weights of the network
        """
        for i, vector in enumerate(weight_adjustments):
            self.__weights[i] += vector

    def adjust_biases(self, bias_adjustments: list[numpy.ndarray]) -> None:
        """
        adjust the biases of the network
        """
        for i, vector in enumerate(bias_adjustments):
            self.__biases[i] += vector
