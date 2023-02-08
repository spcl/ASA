# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical expansion for Reshaping the output of im2col convolution to match B,C,H,W
"""

from dace.transformation.interstate import InlineSDFG

import numpy as np
import dace
from dace.transformation.interstate import InlineSDFG

#imports from libnode expansions
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties
from dace import library


@dace.library.expansion
class ExpandIm2colReshapeOutput(ExpandTransformation):
    '''
    Reshapes the output of im2col convolution using maps
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        Y_col = in_desc_with_name(node, parent_state, parent_sdfg, "Y_col")
        Y = out_desc_with_name(node, parent_state, parent_sdfg, "Y")

        B_S = Y.shape[0]  #batch size
        O_C = Y.shape[1]
        F_H = node.filter_h
        F_W = node.filter_w
        output_height = Y.shape[2]  #((H - (F_H)) + 1)
        output_width = Y.shape[3]  #((W - (F_W)) + 1)

        assert node.padding == 0  # Otherwise we need to add the conditional read in the tasklet
        assert node.stride == 1

        #### Add containers
        _, array_y_col = sdfg.add_array("Y_col", Y_col.shape, dtype=Y_col.dtype)
        _, array_y = sdfg.add_array("Y", Y.shape, dtype=Y.dtype)

        # @dace.program
        # def add_bias(Y_col, Y_col_bias, Bias):
        #     for i in dace.map[0:Y_col.shape[0]]:
        #         Y_col_bias[i] += Y_col[i] + Bias[i]

        # #Note: it's a sort of blackmagic how is working here, the point is that the name of the arrays should match
        # program = add_bias.to_sdfg(array_y_col, array_y_col_bias, array_bias)
        # program.view()

        y_col_read = state.add_read("Y_col")
        y_write = state.add_write("Y")

        im2col_me, im2col_mx = state.add_map(
            "im2col_bias_map",
            {
                "b": f"0:{B_S}",
                "f": f"0:{O_C}",  # num_filters
                "h": f"0:{output_height}",
                "w": f"0:{output_width}",
            })
        tasklet = state.add_tasklet("copy_tasklet", {"in_y"}, {"out_y"}, "out_y = in_y")

        # add memlets
        state.add_memlet_path(y_col_read,
                              im2col_me,
                              tasklet,
                              dst_conn="in_y",
                              memlet=dace.Memlet(f"Y_col[f, b*{output_height*output_width} + h*{output_width} + w]"))
        state.add_memlet_path(tasklet, im2col_mx, y_write, src_conn="out_y", memlet=dace.Memlet("Y[b,f,h,w]"))
        return sdfg


@dace.library.node
class Im2colReshapeOutput(dace.sdfg.nodes.LibraryNode):
    '''
        Rshape the output of im2col convolution
        - The input is O_C x (O_H *O_W * B_S)
        - the output is B_S x O_C x O_h x O_W
        
    '''

    # Global properties
    implementations = {"default": ExpandIm2colReshapeOutput}
    default_implementation = "default"

    # Object fields
    stride = properties.SymbolicProperty(allow_none=False, default=1)
    padding = properties.SymbolicProperty(allow_none=False, default=0)
    filter_h = properties.SymbolicProperty(allow_none=False, default=0)
    filter_w = properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, filter_h, filter_w, location=None, stride=1, padding=0):
        super().__init__(name, location=location, inputs={"Y_col"}, outputs={"Y"})

        self.stride = stride
        self.padding = padding
        self.filter_h = filter_h
        self.filter_w = filter_w

    def validate(self, sdfg, state):
        #TODO
        pass


###########################################################################
# End of library node
###########################################################################
