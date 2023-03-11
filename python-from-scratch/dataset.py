import csv

import numpy

class DataSet:
    """
    data set class for training neural networks
    stores inputs and the expected outputs for those inputs
    reads data from a csv file of this format:
    inputs;delimited;by;semicolon
    more;inputs;semicolon;delimited
    outputs;delimited;by;semicolon
    more;outputs;semicolon;delimited
    """

    def __init__(self, path: str):
        self.__load_data(path)

    def __load_data(self, path: str) -> None:
        """
        load training data from path
        """
        data: list[numpy.ndarray] = []
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=";")
            for row in reader:
                data.append(numpy.array([float(n) for n in row[:-1]]))
        halfway = len(data) // 2
        self.__inputs = numpy.array(data[:halfway])
        self.__outputs = numpy.array(data[halfway:])

    def get_size(self) -> int:
        """
        returns the total number of input-output pairs
        """
        return len(self.__outputs)

    def get_input(self, index: int) -> numpy.ndarray:
        """
        returns a specific input
        """
        return self.__inputs[index]

    def get_output(self, index: int) -> numpy.ndarray:
        """
        returns a specific output
        """
        return self.__outputs[index]

    def get_inputs(self) -> numpy.ndarray:
        """
        returns all inputs
        """
        return self.__inputs

    def get_outputs(self) -> numpy.ndarray:
        """
        returns all outputs
        """
        return self.__outputs
