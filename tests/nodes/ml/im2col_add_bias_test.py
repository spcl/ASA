# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Tests for im2col_add_bias expansion
"""

import numpy as np
import dace

from canonical_libnodes.ml.im2col_add_bias import Im2colAddBias
from dace.transformation.interstate import InlineSDFG


def make_sdfg_libnode(Y_col, Bias):
    # version with libnode

    sdfg = dace.SDFG("im2col_add_bias")

    state = sdfg.add_state("im2col_add_bias_state")

    #### Add containers
    _, array_y_col = sdfg.add_array("Y_col", Y_col.shape, dtype=np.float32)
    _, array_y_col_bias = sdfg.add_array("Y_col_bias", Y_col.shape, dtype=np.float32)
    _, array_bias = sdfg.add_array("Bias", Bias.shape, dtype=np.float32)

    y_col_read = state.add_read("Y_col")
    bias_read = state.add_read("Bias")
    y_col_bias_write = state.add_write("Y_col_bias")

    # add libnode
    libnode = Im2colAddBias('Im2colAddBias')

    state.add_node(libnode)

    state.add_edge(y_col_read, None, libnode, 'Y_col', dace.Memlet.from_array("Y_col", array_y_col))
    state.add_edge(bias_read, None, libnode, 'Bias', dace.Memlet.from_array("Bias", array_bias))
    state.add_edge(libnode, 'Y_col_bias', y_col_bias_write, None,
                   dace.Memlet.from_array("Y_col_bias", array_y_col_bias))
    return sdfg


def run(
    in_channels,
    out_channels,
    kernel_size,
    data_shape: tuple,
    input_to_constant: bool = False,
):

    #create data
    # fake input matrix and bias
    Y_col = np.random.rand(out_channels, data_shape[2] * data_shape[3]).astype(np.float32)
    Bias = np.random.rand(out_channels).astype(np.float32)

    num_filters = out_channels
    num_channels = in_channels
    filter_h = kernel_size
    filter_w = kernel_size

    Y_col_bias = np.empty(Y_col.shape, dtype=np.float32)
    for i in range(Y_col.shape[0]):
        Y_col_bias[i] = Y_col[i] + Bias[i]

    sdfg = make_sdfg_libnode(Y_col, Bias)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    # Build the matrices
    dace_Y_col_bias = np.empty(Y_col.shape, dtype=np.float32)
    sdfg(Y_col=Y_col, Bias=Bias, Y_col_bias=dace_Y_col_bias)
    assert np.allclose(Y_col_bias, dace_Y_col_bias)


def test_im2col_add_bias():
    run(1, 6, 5, (10, 1, 28, 28))


if __name__ == "__main__":

    # Example: second convolutional layer in Lenet
    run(1, 6, 5, (10, 1, 28, 28))