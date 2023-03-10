#!/usr/bin/python3

from network import Network

import numpy

network_test = Network([3, 9999, 2])

print(network_test.calculate_outputs(numpy.random.randn(3)))
