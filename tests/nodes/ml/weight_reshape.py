"""
Reshapes the input features as a matrix using im2col approach
"""

## Needs DaCe v0.13.3

from dace.transformation.interstate import InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from daceml.torch import DaceModule, dace_module
import dace
import argparse
# from daceml.util import utils
from dace.transformation.interstate import InlineSDFG

from dace.transformation.interstate import FPGATransformSDFG

import torch
import torch.nn as nn
import argparse
import numpy as np

import pytest
import dace
# from daceml.util import utils
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG, InlineMultistateSDFG, StateFusion

#imports from libnode expansions
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
import copy

B_S, C, H, W, O_C, F_H, F_W = (dace.symbol(s, dace.int32) for s in ('B_S', 'C', 'H', 'W', 'O_C', 'F_H', 'F_W'))


@dace.program
def dace_weight_reshape(W: dace.float32[O_C, C, F_H, F_W]):

    return np.reshape(W,  [O_C, C * F_H * F_W])

  

def run(
    in_channels,
    out_channels,
    kernel_size,
    data_shape: tuple,
    input_to_constant: bool = False,
):

    #create data
    W = np.random.rand(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)

    num_filters = out_channels
    num_channels = in_channels
    filter_h = kernel_size
    filter_w = kernel_size

    # TODO
    W_col = W.reshape(num_filters, num_channels * filter_h * filter_w)

    B_S =data_shape[0]
    C = num_channels
    H = data_shape[2]
    W = data_shape[3]
    O_C = out_channels
    F_H = filter_h
    F_W = filter_w
    sdfg = dace_weight_reshape.to_sdfg()
    sdfg.view()

    dace_W = sdfg(W)
    assert np.allclose(W_col, dace_W)
    # new_output = np.dot(W_col, i2c)


if __name__ == "__main__":

    # Example: second convolutional layer in Lenet
    run(1, 6, 5, (10, 1, 28, 28))