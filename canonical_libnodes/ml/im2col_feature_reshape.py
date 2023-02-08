# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical Expansion for Reshaping the input features as a matrix using im2col approach
"""

import numpy as np
import dace
from dace.transformation.interstate import InlineSDFG
from dace import library
#imports from libnode expansions
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties


@dace.library.expansion
class ExpandIm2colFeatureReshape(ExpandTransformation):
    '''
    Reshapes the input features according to im2col
    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")

        # Get descriptors (note this may have a shape that is not the one where we expect. E.g., a view on a higher order container
        # will show here the original shape of the container, not the viewed one. In that case use something like _get_mamul_operator)
        X = in_desc_with_name(node, parent_state, parent_sdfg, "X")
        Y = out_desc_with_name(node, parent_state, parent_sdfg, "Y")

        B_S = X.shape[0]  #batch size
        C = X.shape[1]  # input channels
        H = X.shape[2]
        W = X.shape[3]
        F_H = node.filter_h
        F_W = node.filter_w
        output_height = ((H - (F_H)) + 1)
        output_width = ((W - (F_W)) + 1)
        offset = 2 * (node.filter_h // 2 - node.padding)

        assert node.padding == 0  # Otherwise we need to add the conditional read in the tasklet

        state = sdfg.add_state(node.label + "_state")
        #### Add containers
        _, array_x = sdfg.add_array("X", X.shape, dtype=X.dtype)
        _, array_y = sdfg.add_array("Y", Y.shape, dtype=Y.dtype)

        x_read = state.add_read("X")
        y_write = state.add_write("Y")

        im2col_me, im2col_mx = state.add_map(
            "im2col_map",
            {
                "b": f"0:{B_S}",
                "c": f"0:{C}",  # repeat B for computing the result
                "f_h": f"0:{F_H}",
                "f_w": f"0:{F_W}",
                "h": f"0:{output_height}",
                "w": f"0:{output_width}",
            })
        tasklet = state.add_tasklet("read_X", {"from_x"}, {"to_y"}, "to_y = from_x")

        # add memlets
        state.add_memlet_path(x_read,
                              im2col_me,
                              tasklet,
                              dst_conn="from_x",
                              memlet=dace.Memlet("X[b, c, h + f_h, w + f_w]"))
        state.add_memlet_path(
            tasklet,
            im2col_mx,
            y_write,
            src_conn="to_y",
            memlet=dace.Memlet(
                f"Y[c * ({F_H} * {F_W}) + f_h * {F_W} + f_w, b * ({output_width} * {output_height}) + h * {output_width} + w]"
            ))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandIm2colFeatureReshape.make_sdfg(node, state, sdfg)


@dace.library.node
class Im2colFeatureReshape(dace.sdfg.nodes.LibraryNode):
    '''
        Reshape input features according to the im2col approach

        The result will be a matrix having size NXM, where
        - N = input_channels*filter_h*filter_w
        - M = output_width,*output_height * NumBatches
    '''

    # Global properties
    implementations = {"default": ExpandIm2colFeatureReshape}
    default_implementation = "default"

    # Object fields

    stride = properties.SymbolicProperty(allow_none=False, default=1)
    padding = properties.SymbolicProperty(allow_none=False, default=0)
    filter_h = properties.SymbolicProperty(allow_none=False, default=0)
    filter_w = properties.SymbolicProperty(allow_none=False, default=0)

    def __init__(self, name, filter_h, filter_w, location=None, stride=1, padding=0):
        super().__init__(name, location=location, inputs={"X"}, outputs={"Y"})
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
