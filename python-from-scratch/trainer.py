from network import Network
from dataset import DataSet

import numpy
import scipy


class Trainer:
    """
    neural network trainer class
    using back-propagation to calculate the necessary changes
    """

    def __init__(self, network: Network, dataset: DataSet):
        self.__network = network
        self.__dataset = dataset
        self.__learning_rate = 0.5

    @property
    def __outputs(self) -> numpy.ndarray:
        """
        output property of the neural network trainer
        returns a 3D numpy matrix of all the outputs for all the inputs
        """
        return numpy.array(
            [
                self.__network.calculate_output(self.__dataset.get_input(i))
                for i in range(self.__dataset.get_size())
            ]
        )

    def calculate_loss(self) -> float:
        """
        calculate the mean squared sum loss of the network
        """
        return numpy.sqrt(((self.__dataset.get_outputs() - self.__outputs) ** 2).mean())

    def back_propagate_on_all_data(self):
        """
        create new neural network based on the results of back propagation
        """
        layer_sizes = self.__network.get_layer_sizes()
        layer_count = len(layer_sizes)
        self.__weight_adjustment_array = [
            numpy.array(
                [
                    [
                        0.0
                        for k in range(previous_layer_size)
                    ]
                    for j in range(layer_sizes[i + 1])
                ]
            )
            for i, previous_layer_size in enumerate(layer_sizes[:-1])
        ]
        self.__bias_adjustment_array = [
            numpy.array(
                [
                    0.0
                    for i in range(layer_size)
                ]
            )
            for layer_size in layer_sizes[1:]
        ]

        dataset_size = self.__dataset.get_size()
        for i in range(dataset_size):
            self.back_propagate_on_data(i)
        self.__weight_adjustment_array = [
            (vector / dataset_size) * self.__learning_rate
            for vector in self.__weight_adjustment_array
        ]
        self.__bias_adjustment_array = [
            (vector / dataset_size) * self.__learning_rate
            for vector in self.__bias_adjustment_array
        ]

    def back_propagate_on_data(self, index) -> None:
        """
        get back propagation values for 1 data input
        """
        layer_sizes = self.__network.get_layer_sizes()
        layer_count = len(layer_sizes)

        network_state = self.__network.get_state(self.__dataset.get_input(index))

        virtual_value_adjustment_array = [
            numpy.array(
                [
                    0.0
                    for i in range(layer_size)
                ]
            )
            for layer_size in layer_sizes
        ]

        virtual_value_adjustment_array[-1] = self.__dataset.get_output(index) - network_state[-1]
        self.__bias_adjustment_array[-1] += virtual_value_adjustment_array[-1]
        # self.__weight_adjustment_array[-1] += virtual_value_adjustment_array[-1] * numpy.linalg.lstsq(self.__network.get_weights(-1), network_state[-1])

        # virtual_value_adjustment_array[-2] = numpy.linalg.solve(self.__network.get_weights(-1), (virtual_value_adjustment_array[-1] - self.__network.get_biases(index)))
        # self.__bias_adjustment_array[-2] += virtual_value_adjustment_array[-2]
        # self.__weight_adjustment_array[-2] += virtual_value_adjustment_array[-2] * numpy.linalg.solve(self.__network.get_weights(-2), network_state[-2])

    def train(self) -> None:
        """
        train the network by 1 generation
        """
        self.back_propagate_on_all_data()
        self.__network.adjust_weights(self.__weight_adjustment_array)
        self.__network.adjust_biases(self.__bias_adjustment_array)
