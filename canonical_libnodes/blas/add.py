# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Canonical expansion for a library node that adds a constant to its input data
'''

import dace
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
import copy
from dace import library


@dace.library.expansion
class ExpandAddConstant(ExpandTransformation):
    '''
    Add one
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        var = node._constant

        @dace.program
        def add(_in, _out):
            _out[:] = _in + var

        in_desc = in_desc_with_name(node, parent_state, parent_sdfg, "_in")
        out_desc = out_desc_with_name(node, parent_state, parent_sdfg, "_out")
        # # TODO: generalize to multidimension

        _, array_in = sdfg.add_array("_in",
                                     in_desc.shape,
                                     in_desc.dtype,
                                     strides=in_desc.strides,
                                     storage=in_desc.storage)
        _, array_out = sdfg.add_array("_out",
                                      out_desc.shape,
                                      out_desc.dtype,
                                      strides=out_desc.strides,
                                      storage=out_desc.storage)
        program = add.to_sdfg(array_in, array_out)

        return program


@dace.library.node
class AddConstant(dace.sdfg.nodes.LibraryNode):
    '''
        ADD by one
    '''

    # Global properties
    implementations = {
        "pure": ExpandAddConstant,
    }
    default_implementation = "pure"

    def __init__(self, name, constant=1, location=None):
        super().__init__(name, location=location, inputs={"_in"}, outputs={"_out"})
        self._constant = constant

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [1]:
            raise ValueError("Expected 1 inputs to Gather")

        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one output from outer product")
        in_memlet = in_edges[0].data

        in_subset = copy.deepcopy(in_memlet.subset)
        in_subset.squeeze()
        size_in = in_subset.size()

        if len(size_in) != 2:
            raise NotImplementedError("Add currenly support 2D shapes")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from outer product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()
        if (len(size_out) != 2 or size_out[0] != size_in[0] or size_out[1] != size_in[1]):
            raise ValueError("Output shape of Gather does not match with input shape.")


# Replacement from function call to libnode
from dace.frontend.common.op_repository import replaces


@replaces('AddConstant')
def add_libnode(pv: 'ProgramVisitor', sdfg: dace.SDFG, state: dace.SDFGState, inp, outp, constant):
    # create the libnode and connect it
    rin = state.add_read(inp)
    wout = state.add_write(outp)
    libnode = AddConstant('Add', constant=constant)
    state.add_node(libnode)

    state.add_edge(rin, None, libnode, '_in', dace.Memlet(inp))
    state.add_edge(libnode, '_out', wout, None, dace.Memlet(outp))

    return []
