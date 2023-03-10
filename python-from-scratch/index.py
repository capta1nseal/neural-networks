#!/usr/bin/python3

from network import Network

import numpy

network_test = Network([3, 5, 2])

for i in range(999):
    print(network_test.calculate_outputs(numpy.random.randn(3)))
