# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical Expansion for adding Bias to the computed im2col matrix
"""

import dace
from dace import library
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name


@dace.library.expansion
class ExpandIm2colAddBias(ExpandTransformation):
    '''
    Reshapes the input weights according to im2col
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        Y_col = in_desc_with_name(node, parent_state, parent_sdfg, "Y_col")
        Y_col_bias = out_desc_with_name(node, parent_state, parent_sdfg, "Y_col_bias")
        Bias = in_desc_with_name(node, parent_state, parent_sdfg, "Bias")
        H = Y_col.shape[0]
        W = Y_col.shape[1]

        #### Add containers
        _, array_y_col = sdfg.add_array("Y_col", Y_col.shape, dtype=Y_col.dtype)
        _, array_y_col_bias = sdfg.add_array("Y_col_bias", Y_col_bias.shape, dtype=Y_col_bias.dtype)
        _, array_bias = sdfg.add_array("Bias", Bias.shape, dtype=Bias.dtype)

        # @dace.program
        # def add_bias(Y_col, Y_col_bias, Bias):
        #     for i in dace.map[0:Y_col.shape[0]]:
        #         Y_col_bias[i] += Y_col[i] + Bias[i]

        # #Note: it's a sort of magic how is working here, the point is that the name of the arrays should match
        # program = add_bias.to_sdfg(array_y_col, array_y_col_bias, array_bias). # Better to do it manually

        y_col_read = state.add_read("Y_col")
        bias_read = state.add_read("Bias")
        y_col_bias_write = state.add_write("Y_col_bias")

        im2col_me, im2col_mx = state.add_map("im2col_bias_map", {
            "h": f"0:{H}",
            "w": f"0:{W}",
        })
        tasklet = state.add_tasklet("add_bias", {"in_y", "in_bias"}, {"out_y"}, "out_y = in_y + in_bias")

        # add memlets
        state.add_memlet_path(y_col_read, im2col_me, tasklet, dst_conn="in_y", memlet=dace.Memlet("Y_col[h, w]"))
        state.add_memlet_path(bias_read, im2col_me, tasklet, dst_conn="in_bias", memlet=dace.Memlet("Bias[h]"))
        state.add_memlet_path(tasklet,
                              im2col_mx,
                              y_col_bias_write,
                              src_conn="out_y",
                              memlet=dace.Memlet(f"Y_col_bias[h,w]"))
        return sdfg


@dace.library.node
class Im2colAddBias(dace.sdfg.nodes.LibraryNode):
    '''
        Adds the bias, row by row in the computed matrix
        
    '''

    # Global properties
    implementations = {"default": ExpandIm2colAddBias}
    default_implementation = "default"

    # Object fields

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs={"Y_col", "Bias"}, outputs={"Y_col_bias"})

    def validate(self, sdfg, state):
        #TODO
        pass


###########################################################################
# End of library node
###########################################################################
