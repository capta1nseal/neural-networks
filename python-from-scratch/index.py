#!/usr/bin/python3

from network import Network
from dataset import DataSet
from trainer import Trainer

import numpy

network = Network([3, 5, 2])
dataset = DataSet("sample_data_3_to_2")
trainer = Trainer(network, dataset)

print(network.calculate_output(numpy.array([1.0, 0.5, 0.25])))

for i in range(999):
    trainer.train()
    print(network.calculate_output(numpy.array([1.0, 0.5, 0.25])))
