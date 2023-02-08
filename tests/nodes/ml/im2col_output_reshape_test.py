# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
 Test for Reshaping the output result of im2col convolution
"""

## Needs DaCe v0.13.3

import numpy as np
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes.ml.im2col_output_reshape import Im2colReshapeOutput
import dace


def make_sdfg_libnode(Y_col, Y, filter_h, filter_w):
    # version with libnode

    sdfg = dace.SDFG("im2col_add_bias")

    state = sdfg.add_state("im2col_add_bias_state")

    #### Add containers
    _, array_y_col = sdfg.add_array("Y_col", Y_col.shape, dtype=np.float32)
    _, array_y = sdfg.add_array("Y", Y.shape, dtype=np.float32)

    y_col_read = state.add_read("Y_col")
    y_write = state.add_write("Y")

    # add libnode
    libnode = Im2colReshapeOutput('Im2ColReshapeOutput', filter_h, filter_w)

    state.add_node(libnode)

    state.add_edge(y_col_read, None, libnode, 'Y_col', dace.Memlet.from_array("Y_col", array_y_col))
    state.add_edge(libnode, 'Y', y_write, None, dace.Memlet.from_array("Y", array_y))
    return sdfg


def run(
    in_channels,
    out_channels,
    kernel_size,
    data_shape: tuple,
    input_to_constant: bool = False,
):

    #create data
    # fake input matrix
    B = data_shape[0]
    C = data_shape[1]
    filter_h = kernel_size
    filter_w = kernel_size
    padding = 0
    stride = 1
    output_height = int((data_shape[2] + (2 * padding) - (filter_h)) / stride) + 1
    output_width = int((data_shape[3] + (2 * padding) - (filter_w)) / stride) + 1

    Y_col = np.random.rand(out_channels, B * output_height * output_width).astype(np.float32)
    Y_ref = np.random.rand(B, out_channels, output_height, output_width).astype(np.float32)
    Y_dace = np.random.rand(B, out_channels, output_height, output_width).astype(np.float32)

    for b in range(0, B):  # batch
        for f in range(0, out_channels):
            for xx in range(0, output_height):
                for yy in range(0, output_width):
                    Y_ref[b, f, xx, yy] = Y_col[f, b * (output_height * output_width) + xx * output_width + yy]

    sdfg = make_sdfg_libnode(Y_col, Y_dace, filter_h, filter_w)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    # Build the matrices
    sdfg(Y_col=Y_col, Y=Y_dace)
    assert np.allclose(Y_dace, Y_ref)


def test_im2col_output_reshape():
    run(1, 6, 5, (10, 1, 28, 28))


if __name__ == "__main__":
    # Example: second convolutional layer in Lenet
    run(1, 6, 5, (10, 1, 28, 28))