# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    In this sample, the application consist of a pipeline og Convolution, MaxPool and Relu operators,
    which is common to many Convolutional Neural Networks.

    The SDFG is extracted by using DaCeML.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.torch import DaceModule
import dace
import daceml
from dse.dse import DSE
from dse.analysis import get_pareto_configurations


### Model definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 6, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        return x


if __name__ == "__main__":

    data_shape = (1, 1, 16, 16)

    # To generate the application SDFG, we need to use DaCeML. This requires
    # to first create a pytorch model, and then parse it using DaCeML

    ptmodel = Model()

    #create data
    x = torch.rand(data_shape)

    # Create SDFG
    dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)
    sdfg = dace_model.sdfg
    from dace.transformation.dataflow import RedundantArray
    sdfg.apply_transformations_repeated([RedundantArray], print_report=True)

    # change name to SDFG (as it will be used to save DSE results into csv file)
    sdfg.name = "ml_simple"

    # Run DSE
    results = DSE(
        sdfg,
        num_pes=[32, 64, 128],
        on_chip_memory_sizes=[256, 1024],
        use_multithreading=True,  # use multithreading
        n_threads=8)

    # Get pareto frontier configurations
    get_pareto_configurations(results, sdfg.name)
