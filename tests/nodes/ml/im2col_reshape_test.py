# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
 Test for Reshaping the input features as a matrix using im2col approach
"""

## Needs DaCe v0.13.3

import numpy as np
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes.ml.im2col_feature_reshape import Im2colFeatureReshape
import dace


def make_sdfg_libnode(X, out_channels, filter_h, filter_w, padding, stride):
    # version with libnode
    B_S = X.shape[0]  #batch size
    C = X.shape[1]  # input channels
    H = X.shape[2]
    W = X.shape[3]
    O_C = out_channels
    F_H = filter_h
    F_W = filter_w
    output_height = ((H - (F_H)) + 1)
    output_width = ((W - (F_W)) + 1)
    offset = 2 * (filter_h // 2 - padding)

    assert padding == 0  # Otherwise we need to add the conditional read in the tasklet

    sdfg = dace.SDFG("im2col_reshape")

    state = sdfg.add_state("im2col_reshape_state")

    #### Add containers
    _, array_x = sdfg.add_array("X", X.shape, dtype=np.float32)
    _, array_y = sdfg.add_array("Y", (C * F_H * F_W, ((H - (F_H)) + 1) * ((W - (F_W)) + 1) * B_S), dtype=np.float32)

    x_read = state.add_read("X")
    y_write = state.add_write("Y")

    # add libnode
    libnode = Im2colFeatureReshape('Im2colReshape', filter_h=filter_h, filter_w=filter_w)

    ##### ADD custom expansion to existing Library Node
    #Gemm.register_implementation("my_own", ExpandMMMPure)

    state.add_node(libnode)

    state.add_edge(x_read, None, libnode, 'X', dace.Memlet.from_array("X", array_x))
    state.add_edge(libnode, 'Y', y_write, None, dace.Memlet.from_array("Y", array_y))

    return sdfg


def my_im2col(X, filter_h, filter_w, padding, stride):
    # Manual implementation of the reshaping
    # The result will be a matrix having size NXM, where
    # - N = input_channels*filter_h*filter_w
    # - M = output_width,*output_height * NumBatches

    B = X.shape[0]
    C = X.shape[1]
    output_height = int((X.shape[2] + (2 * padding) - (filter_h)) / stride) + 1
    output_width = int((X.shape[3] + (2 * padding) - (filter_w)) / stride) + 1

    im2col_vector = np.zeros((C * filter_h * filter_w, output_height * output_width * B))
    # offset = 2 * (filter_h // 2 - padding)
    # TODO: which of the two is correct?
    offset = filter_h
    print("Output size: ", output_height, output_width)
    for b in range(0, B):
        for c in range(0, C):
            for f_h in range(0, filter_h):
                for f_w in range(0, filter_w):
                    for x in range(0, output_height):
                        for y in range(0, output_width):
                            if (x + f_h - padding < output_height + offset) and (x + f_h - padding >= 0) and (
                                    y + f_w - padding < output_width + offset) and (y + f_w - padding >= 0):
                                data = X[b, c, x + f_h, y + f_w]
                            else:
                                data = 0

                            im2col_vector[c * (filter_h * filter_w) + f_h * filter_w + f_w,
                                          b * (output_width * output_height) + x * output_width + y] = data

    return im2col_vector


def run(
    in_channels,
    out_channels,
    kernel_size,
    data_shape: tuple,
    input_to_constant: bool = False,
):

    #create data
    x = np.random.rand(*data_shape).astype(np.float32)

    num_filters = out_channels
    num_channels = in_channels
    filter_h = kernel_size
    filter_w = kernel_size

    padding = 0
    stride = 1

    i2c = my_im2col(x, filter_h, filter_w, padding, stride)

    B_S = x.shape[0]
    C = num_channels
    H = x.shape[2]
    W = x.shape[3]
    O_C = out_channels
    F_H = filter_h
    F_W = filter_w
    sdfg = make_sdfg_libnode(x, out_channels, filter_h, filter_w, padding, stride)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

    Y = np.empty((C * F_H * F_W, ((H - (F_H)) + 1) * ((W - (F_W)) + 1) * B_S), dtype=np.float32)
    sdfg(X=x, Y=Y)
    assert np.allclose(i2c, Y)


def test_im2col_feature_reshape():
    run(1, 6, 5, (10, 1, 28, 28))


if __name__ == "__main__":

    # Example: second convolutional layer in Lenet
    run(1, 6, 5, (10, 1, 28, 28))