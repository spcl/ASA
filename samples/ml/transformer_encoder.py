# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    In this sample, the application consist of a transformer encoder layer.

    The SDFG is extracted by using DaCeML.

    The model can be generated in a smaller version (setting small to True) for
    a quicker DSE (1 hour on a 16 cores machine).
    The base transformer encoder layer of "Attention is all you need" requires
    ca. 12 hours for full DSE (on a 16 cores machine)/

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.torch import DaceModule
import daceml.onnx as donnx
from dse.dse import DSE
from dse.analysis import get_pareto_configurations

if __name__ == "__main__":
    '''
    This function is used to evaluate a given model.
    It will build the pytorch model, transform it to a DaCe Model, apply transformation and execute on FPGA
    :return: returns if the result is correct
    '''
    # create pytorch model and data

    small = False

    if small:
        ptmodel = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=32)
        src = torch.rand(10, 32, 128)

    else:
        ptmodel = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
        src = torch.rand(10, 32, 512)

    # Create the SDFG using DaCeML
    import daceml.onnx as donnx
    old_default = donnx.default_implementation

    donnx.default_implementation = "pure"

    dace_model = DaceModule(ptmodel, dummy_inputs=(src, ), auto_optimize=True)  # fails here inside daceml

    sdfg = dace_model.sdfg
    from dace.transformation.dataflow import RedundantArray
    sdfg.apply_transformations_repeated([RedundantArray], print_report=True)

    # Perform DSE
    results = DSE(sdfg,
                  num_pes=[64, 128, 256, 512, 1024, 2048],
                  use_multithreading=True,
                  n_threads=16,
                  on_chip_memory_sizes={256, 1024, 8192})
    get_pareto_configurations(results, sdfg.name)
