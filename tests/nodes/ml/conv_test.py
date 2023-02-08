# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Test for Im2Col canonical expansion of Convolution
"""

## Needs DaCe v0.13.3

from dace.transformation.interstate import InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.torch import DaceModule, dace_module
import dace
import argparse
from daceml.util import utils

import daceml

from canonical_libnodes.ml.im2col_conv import ExpandConvCanonical

B_S, C, H, W, O_C, F_H, F_W = (dace.symbol(s, dace.int32) for s in ('B_S', 'C', 'H', 'W', 'O_C', 'F_H', 'F_W'))


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_to_constant):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        if input_to_constant:
            #fix the weight otherwise everytime they are randomized
            self.conv.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(1)

    def forward(self, x):
        return self.conv(x)


def run(
    in_channels,
    out_channels,
    kernel_size,
    data_shape: tuple,
    input_to_constant: bool = False,
):
    '''
    This function is used to evaluate a given model.
    It will build the pytorch model, transform it to a DaCe Model, apply transformation and execute on FPGA
    :return: returns if the result is correct
    '''
    # create pytorch model
    ptmodel = Model(in_channels, out_channels, kernel_size, input_to_constant)

    #create data
    x = torch.rand(data_shape)

    #evaluate pytorch model
    torch_output = ptmodel(x)

    # #create dace model
    daceml.onnx.nodes.onnx_op.ONNXConv.register_implementation("canonical", ExpandConvCanonical)

    with dace.library.change_default(daceml.onnx.nodes.onnx_op.ONNXConv, "canonical"):
        dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)

    dace_output = dace_model(x)
    sdfg = dace_model.sdfg

    ## Need to remove redundant Array in newer DaCeML versions
    from dace.transformation.dataflow import RedundantArray
    sdfg.apply_transformations_repeated([RedundantArray], print_report=True)

    sdfg.expand_library_nodes(recursive=False)
    sdfg.apply_transformations_repeated([InlineSDFG()], print_report=True)
    # sdfg.view()
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


def test_conv():
    run(1, 2, 3, (1, 1, 8, 8))


if __name__ == "__main__":

    # Lenet second conv
    # run(1, 6, 5, (1, 1, 28, 28))
    # Small Conv
    run(1, 2, 3, (1, 1, 8, 8))