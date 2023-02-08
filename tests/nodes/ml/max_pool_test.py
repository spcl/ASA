# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.

## Needs DaCe 0.13.3

from dace.transformation.interstate import InlineSDFG
import daceml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.torch import DaceModule, dace_module
import dace
import argparse
from daceml.util import utils
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes.ml.maxpool import ExpandMaxPoolCanonical


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 2)


def run(data_shape: tuple, ):
    '''
    Evaluates a specific configuration
    :param data_shape:
    :return:
    '''

    ptmodel = Model()
    x = torch.rand(data_shape) - 0.5
    torch_output = ptmodel(x)

    daceml.onnx.nodes.onnx_op.ONNXMaxPool.register_implementation("canonical", ExpandMaxPoolCanonical)

    with dace.library.change_default(daceml.onnx.nodes.onnx_op.ONNXMaxPool, "canonical"):
        dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)
    sdfg = dace_model.sdfg
    # This is needed in case you want to convert it to canonical DAG

    dace_output = dace_model(x)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

    ref = torch_output.detach().numpy()
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


def test_maxpool():
    run((10, 6, 32, 32))


if __name__ == "__main__":

    run((10, 6, 32, 32))