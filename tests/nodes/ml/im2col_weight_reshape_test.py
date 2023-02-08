# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
 Test for Reshaping the input weights for im2col convolution
"""

## Needs DaCe v0.13.3

import numpy as np
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes.ml.im2col_weight_reshape import Im2colWeightReshape
import dace


def make_sdfg_libnode(W):

    # version with libnode
    O_C = W.shape[0]
    C = W.shape[1]
    F_H = W.shape[2]
    F_W = W.shape[3]

    sdfg = dace.SDFG("im2col_reshape")

    state = sdfg.add_state("im2col_reshape_state")

    #### Add containers
    _, array_w = sdfg.add_array("W", W.shape, dtype=np.float32)
    _, array_w_col = sdfg.add_array("W_col", [O_C, C * F_H * F_W], dtype=np.float32)

    w_read = state.add_read("W")
    w_col_write = state.add_write("W_col")

    # add libnode
    libnode = Im2colWeightReshape('Im2colWReshape')

    state.add_node(libnode)

    state.add_edge(w_read, None, libnode, 'W', dace.Memlet.from_array("W", array_w))
    state.add_edge(libnode, 'W_col', w_col_write, None, dace.Memlet.from_array("W_col", array_w_col))
    return sdfg


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

    W_col = np.reshape(W, (num_filters, num_channels * filter_h * filter_w))

    C = num_channels
    O_C = out_channels
    F_H = filter_h
    F_W = filter_w

    sdfg = make_sdfg_libnode(W)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

    # Build the matrices
    dace_W = np.empty([O_C, C * F_H * F_W], dtype=np.float32)
    sdfg(W=W, W_col=dace_W)
    assert np.allclose(W_col, dace_W)


def test_im2col_weight_reshape():
    run(1, 6, 5, (10, 1, 28, 28))


if __name__ == "__main__":

    # Example: second convolutional layer in Lenet
    run(1, 6, 5, (10, 1, 28, 28))
