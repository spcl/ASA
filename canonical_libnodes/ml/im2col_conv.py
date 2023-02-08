# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical Expansion for Im2Col implementation of Convolution.

    This is lowered into a set of composite canonical libnodes:
    1. reshape nodes for the input weights and futures
    2. matrix multiplication for the actual convolution
    3. bias addition 
    4. output reshape

"""

import dace

import dace

#imports from libnode expansions
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties

from canonical_libnodes.ml.im2col_feature_reshape import Im2colFeatureReshape
from canonical_libnodes.ml.im2col_weight_reshape import Im2colWeightReshape
from canonical_libnodes.ml.im2col_add_bias import Im2colAddBias
from canonical_libnodes.ml.im2col_output_reshape import Im2colReshapeOutput
from canonical_libnodes.blas.mmm import MMM
import copy


@dace.library.expansion
class ExpandConvCanonical(ExpandTransformation):
    '''
    This convolution is implemented with the Im2Col approach
    
    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")

        # Get descriptors (note this may have a shape that is not the one where we expect. E.g., a view on a higher order container
        # will show here the original shape of the container, not the viewed one. In that case use something like _get_mamul_operator)
        X = in_desc_with_name(node, parent_state, parent_sdfg, "X")
        W = in_desc_with_name(node, parent_state, parent_sdfg, "W")
        Y = out_desc_with_name(node, parent_state, parent_sdfg, "Y")
        try:
            B = in_desc_with_name(node, parent_state, parent_sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        num_filters = W.shape[0]
        num_channels = X.shape[1]
        batch_size = X.shape[0]
        padding = node.pads[0]  # currently only equal padding supported

        if node.kernel_shape is not None:
            filter_h, filter_w = node.kernel_shape
        else:
            filter_x, filter_w = W.shape[2:]
        offset = 2 * (filter_h // 2 - padding)
        output_size_h, output_size_w = Y.shape[2:]

        ### checks
        # only do 2D for now
        assert len(X.shape) == 4 or len(W.shape) == 4
        assert num_channels == W.shape[1]

        # Equal padding
        assert node.pads is None or (all(p == node.pads[0] for p in node.pads) and not len(node.pads) != image_dims * 2)

        assert node.strides is None or len(node.strides) == image_dims

        ### Build expansion
        state = sdfg.add_state(node.label + "_state")
        sdfg.add_datadesc("X", copy.deepcopy(X))
        sdfg.add_datadesc("W", copy.deepcopy(W))
        sdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            sdfg.add_datadesc("B", copy.deepcopy(B))
            sdfg.arrays["B"].transient = False

        # TODO: is this needed?
        sdfg.arrays["X"].transient = False
        sdfg.arrays["W"].transient = False
        sdfg.arrays["Y"].transient = False

        ## Reshape input data

        ## Weights
        # TODO: understand if we want this to be a transient or not.
        # If this is a transient will try to stream on it (we recognize as buffer node only non transient ones
        # and then we try to apply stremaing transformation). The same is true for the others _col array.
        # But if this is allocated as non transient, then it will not work
        # CURRENT PROVISIONAL FIX: use appendix (_bn)
        _, array_w_col = sdfg.add_array("W_col_bn", [num_filters, num_channels * filter_h * filter_w],
                                        dtype=W.dtype,
                                        transient=True)

        w_read = state.add_read("W")
        w_col = state.add_access("W_col_bn")

        libnode = Im2colWeightReshape('Im2colWReshape')

        state.add_node(libnode)

        state.add_edge(w_read, None, libnode, 'W', dace.Memlet.from_array("W", sdfg.arrays["W"]))
        state.add_edge(libnode, 'W_col', w_col, None, dace.Memlet.from_array("W_col_bn", array_w_col))

        ## Features
        _, array_x_col = sdfg.add_array("X_col_bn", (num_channels * filter_h * filter_w,
                                                     (output_size_h * output_size_w) * batch_size),
                                        dtype=X.dtype,
                                        transient=True)

        x_read = state.add_read("X")
        x_col = state.add_access("X_col_bn")

        libnode = Im2colFeatureReshape('Im2colXReshape', filter_h=filter_h, filter_w=filter_w)

        state.add_node(libnode)

        state.add_edge(x_read, None, libnode, 'X', dace.Memlet.from_array("X", sdfg.arrays["X"]))
        state.add_edge(libnode, 'Y', x_col, None, dace.Memlet.from_array("X_col_bn", array_x_col))

        ## Compute MatMul

        _, array_y_col = sdfg.add_array("Y_col_bn", (num_filters, (output_size_h * output_size_w) * batch_size),
                                        dtype=Y.dtype,
                                        transient=True)

        y_col_access = state.add_access("Y_col_bn")

        libnode = MMM('MMM_im2col', transA=False, transB=False, alpha=1)

        # libnode.implementation = "op_col"  # TODO: change it
        libnode.implementation = "mv"  # TODO: change it
        state.add_node(libnode)

        state.add_edge(w_col, None, libnode, '_a', dace.Memlet.from_array("W_col_bn", array_w_col))
        state.add_edge(x_col, None, libnode, '_b', dace.Memlet.from_array("X_col_bn", array_x_col))
        state.add_edge(libnode, '_c', y_col_access, None, dace.Memlet.from_array("Y_col_bn", array_y_col))

        ## Add Bias
        if B is not None:
            bias_read = state.add_read("B")
            _, array_y_col_bias = sdfg.add_array("Y_col_bias_bn",
                                                 (num_filters, (output_size_h * output_size_w) * batch_size),
                                                 dtype=Y.dtype,
                                                 transient=True)
            y_col_bias_access = state.add_access("Y_col_bias_bn")

            libnode = Im2colAddBias('Im2colAddBias')

            state.add_node(libnode)

            state.add_edge(y_col_access, None, libnode, 'Y_col', dace.Memlet.from_array("Y_col_bn", array_y_col))
            state.add_edge(bias_read, None, libnode, 'Bias', dace.Memlet.from_array("B", sdfg.arrays["B"]))
            state.add_edge(libnode, 'Y_col_bias', y_col_bias_access, None,
                           dace.Memlet.from_array("Y_col_bias_bn", array_y_col_bias))

        ## Reshape to final output
        y_write = state.add_write("Y")

        # add libnode
        libnode = Im2colReshapeOutput('Im2ColReshapeOutput', filter_h, filter_w)

        state.add_node(libnode)

        if B is not None:
            state.add_edge(y_col_bias_access, None, libnode, 'Y_col',
                           dace.Memlet.from_array("Y_col_bias_bn", array_y_col_bias))
        else:
            state.add_edge(y_col_access, None, libnode, 'Y_col', dace.Memlet.from_array("Y_col_bn", array_y_col))
        state.add_edge(libnode, 'Y', y_write, None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandConvCanonical.make_sdfg(node, state, sdfg)