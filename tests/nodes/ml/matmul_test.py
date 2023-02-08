# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
Tests for Matmult
"""

import torch
import torch.nn as nn
import numpy as np
from daceml.torch import DaceModule
import pytest
import dace
import daceml
from dace.transformation.interstate import InlineSDFG

from canonical_libnodes.ml.matmul import ExpandMatmulCanonical


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # equivalent to np.einsum('bik,bkj->bij', A, B)
        z = torch.matmul(x, y)
        return z


def run(x_shape: tuple, y_shape: tuple):
    '''
    Evaluates the given configuration
    :param x_shape:
    :param y_shape:
    :param vec_width:
    :param execute_cpu_dace:
    :param queue:
    :return:
    '''

    import daceml.onnx as donnx

    ptmodel = Model()

    x = torch.ones(x_shape, dtype=torch.float32)
    y = torch.rand(y_shape, dtype=torch.float32)
    torch_output = ptmodel(x, y)

    daceml.onnx.nodes.onnx_op.ONNXMatMul.register_implementation("canonical", ExpandMatmulCanonical)
    with dace.library.change_default(donnx.ONNXMatMul, "canonical"):
        dace_model = DaceModule(ptmodel, dummy_inputs=(
            x,
            y,
        ), auto_optimize=False)
        dace_output = dace_model(x, y)
        sdfg = dace_model.sdfg

    from dace.transformation.dataflow import RedundantArray
    sdfg.apply_transformations_repeated([RedundantArray], print_report=True)

    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


@pytest.mark.parametrize("A_shape, B_shape", [((4, 4, 8), (8, 8)), ((4, 4, 8), (4, 8, 4))])
def test_matmul(A_shape, B_shape):
    run(A_shape, B_shape)


if __name__ == "__main__":
    data_shape_1 = (4, 4, 8)
    data_shape_2 = (4, 8, 4)
    run(data_shape_1, data_shape_2)
