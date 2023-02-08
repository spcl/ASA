# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical expansion for Reshaping the input weight as a matrix using im2col approach
"""

import numpy as np
import dace
from dace.transformation.interstate import InlineSDFG

#imports from libnode expansions
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import library


@dace.library.expansion
class ExpandIm2colWeightReshape(ExpandTransformation):
    '''
    Reshapes the input weights according to im2col
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")

        W = in_desc_with_name(node, parent_state, parent_sdfg, "W")
        Wcol = out_desc_with_name(node, parent_state, parent_sdfg, "W_col")

        #### Add containers
        _, array_w = sdfg.add_array("W", W.shape, dtype=W.dtype)
        _, array_w_col = sdfg.add_array("W_col", Wcol.shape, dtype=Wcol.dtype)

        @dace.program
        def w_reshape(W, W_col):
            W_col[:] = np.reshape(W, W_col.shape)

        #Note: it's a sort of blackmagic how is working here, the point is that the name of the arrays should match
        program = w_reshape.to_sdfg(array_w, array_w_col)

        return program


@dace.library.node
class Im2colWeightReshape(dace.sdfg.nodes.LibraryNode):
    '''
        Reshape input weight according to the im2col approach

        The result will be a matrix having size NXM, where
        - N = output_channels
        - M = input_channels*filter_h*filter_w
    '''

    # Global properties
    implementations = {"default": ExpandIm2colWeightReshape}
    default_implementation = "default"

    # Object fields

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs={"W"}, outputs={"W_col"})
        # self.stride = stride
        # self.padding = padding
        # self.filter_h = filter_h
        # self.filter_w = filter_w

    def validate(self, sdfg, state):
        #TODO
        pass


###########################################################################
# End of library node
###########################################################################
