# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Test for Softmax canonical expansion
"""

## Needs DaCe v0.13.3

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from daceml.torch import DaceModule, dace_module
import dace
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes.ml.softmax import ExpandSoftmaxCanonical
import daceml


class Model(nn.Module):
    def __init__(self, axis):
        super(Model, self).__init__()
        self.axis = axis

    def forward(self, x):
        x = F.softmax(x, dim=self.axis)
        return x


def run(data_shape: tuple, axis, queue=None):

    ptmodel = Model(axis)
    x = torch.rand(data_shape, )

    dace_model = DaceModule(ptmodel, auto_optimize=False)
    import daceml.onnx as donnx

    # register canonical expansion
    daceml.onnx.nodes.onnx_op.ONNXSoftmax.register_implementation("canonical", ExpandSoftmaxCanonical)

    with dace.library.change_default(daceml.onnx.nodes.onnx_op.ONNXSoftmax, "canonical"):
        dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)
        dace_output = dace_model(x)

    torch_output = ptmodel(x)

    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


def test_softmax():
    run((100, 10, 10), 2)


if __name__ == "__main__":

    data_shape = (100, 10, 10)
    run(data_shape, 2)
