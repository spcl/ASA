# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.

## Needs DaCe 0.13.3

# Note: this test is used for analysis purposes (we rely on DaCeML pure expansion for this operation)

from dace.transformation.interstate import InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.torch import DaceModule, dace_module
import dace
import argparse
from daceml.util import utils
from dace.transformation.interstate import InlineSDFG


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.relu(x)


def run(data_shape: tuple, ):
    '''
    Evaluates a specific configuration
    :param data_shape:
    :return:
    '''

    ptmodel = Model()
    x = torch.rand(data_shape) - 0.5
    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import daceml.onnx as donnx
    with dace.library.change_default(donnx.ONNXRelu, "pure"):
        dace_output = dace_model(x)

    torch_output = ptmodel(x)

    sdfg = dace_model.sdfg
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineMultistateSDFG, InlineSDFG], print_report=True)
    sdfg.view()
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


if __name__ == "__main__":
    run((10, 4, 32, 32))