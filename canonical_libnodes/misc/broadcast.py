# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    This is a dummy library node that acts as a broadcaster to favor canonical 
    DAG construction and Streaming analysis.

    The node takes in input a 2D container, produces a container with the same
    shape but data can be read/written either by row or by column

    TODO: generalize to generic shape or axis
'''
import dace
import numpy as np
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
import copy
from dace import library
from dace import properties
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG
from dace.libraries.blas.nodes.matmul import _get_matmul_operands

N, M = (dace.symbol(s, dtype=dace.int32) for s in ('N', 'M'))


@dace.library.expansion
class ExpandBroadcast(ExpandTransformation):
    '''
    "Broadcast" that outputs the content of a container by row or columns depending on the row_major properties.
    NOTE: for the moment being this is just a copy operator for the sake of enabling streaming analyzability at the boundaries
    of canonical library nodes (for streaming composition the produced and consumer map must have the same ranges). 
    It would be cool to implement this as a true broadcast.
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")

        in_desc = in_desc_with_name(node, parent_state, parent_sdfg, "_in")
        out_desc = out_desc_with_name(node, parent_state, parent_sdfg, "_out")
        # # TODO: generalize to multidimension
        # Note: this takes into account if we have to transpose also
        N = out_desc.shape[-2]
        M = out_desc.shape[-1] if len(out_desc.shape) >= 2 else 1
        if node.used_in_batched_matmul:
            BS = out_desc.shape[0]

        sdfg.add_array("_in", in_desc.shape, in_desc.dtype, strides=in_desc.strides, storage=in_desc.storage)
        sdfg.add_array("_out", out_desc.shape, out_desc.dtype, strides=out_desc.strides, storage=out_desc.storage)

        state = sdfg.add_state(node.label + "_state")
        inp = state.add_read("_in")
        outp = state.add_write("_out")
        # Create map, by row
        broadcast_tasklet = state.add_tasklet("broadcast_tasklet", {"_in_val"}, {"_out_val"}, "_out_val = _in_val ")
        # if transpose the map boundaries remain the same (thanks to how N and M are derived)
        # but we need to change the way in which we read from the input container
        if node.row_major:
            assert not node.used_in_batched_matmul
            broadcast_map_entry, broadcast_map_exit = state.add_map("broadcast_map", {'i': f'0:{N}', 'j': f'0:{M}'})
        else:
            if not node.used_in_batched_matmul:
                broadcast_map_entry, broadcast_map_exit = state.add_map("broadcast_map", {'j': f'0:{M}', 'i': f'0:{N}'})
            else:
                broadcast_map_entry, broadcast_map_exit = state.add_map("broadcast_map", {
                    'b': f'0:{BS}',
                    'j': f'0:{M}',
                    'i': f'0:{N}'
                })
        if not node.transpose:
            if not node.used_in_batched_matmul:
                reading_pattern = "_in[i,j]"
            else:
                reading_pattern = "_in[b, i,j]"
        else:
            if not node.used_in_batched_matmul:
                reading_pattern = "_in[j,i]"
            else:
                reading_pattern = "_in[b,j,i]"

        if not node.used_in_batched_matmul:
            writing_pattern = "_out[i,j]"
        else:
            writing_pattern = "_out[b, i,j]"
        state.add_memlet_path(inp,
                              broadcast_map_entry,
                              broadcast_tasklet,
                              dst_conn="_in_val",
                              memlet=dace.Memlet(reading_pattern))
        state.add_memlet_path(broadcast_tasklet,
                              broadcast_map_exit,
                              outp,
                              src_conn="_out_val",
                              memlet=dace.Memlet(writing_pattern))

        return sdfg


@dace.library.node
class Broadcast(dace.sdfg.nodes.LibraryNode):
    '''
        This is a dummy library node that acts as a broadcast to favor canonical 
        DAG construction and Streaming analysis.
    '''

    # Global properties
    implementations = {
        "broadcast": ExpandBroadcast,
    }
    default_implementation = None

    # Object fields
    row_major = properties.Property(dtype=bool,
                                    desc="Whether to produce the data in row major (otherwise column major)",
                                    default=True)
    transpose = properties.Property(dtype=bool, desc="Whether to transpose while broadcasting", default=False)

    used_in_batched_matmul = properties.Property(dtype=bool,
                                                 desc="Whether this is used in a Batched matmul (ONNX node only)")

    def __init__(self, name, location=None, row_major=True, transpose=False, used_in_batched_matmul=False):
        super().__init__(name, location=location, inputs={"_in"}, outputs={"_out"})
        self.row_major = row_major
        self.transpose = transpose
        self.used_in_batched_matmul = used_in_batched_matmul

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [1]:
            raise ValueError("Expected 1 inputs to Broadcast")

        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one output from outer product")
        in_memlet = in_edges[0].data

        in_subset = copy.deepcopy(in_memlet.subset)
        in_subset.squeeze()
        size_in = in_subset.size()

        if not self.used_in_batched_matmul and len(size_in) > 2:
            raise NotImplementedError("Broadcast currently support up to 2D shapes")
        if self.used_in_batched_matmul and len(size_in) > 3:
            raise NotImplementedError("Broadcast with batched matmul currently support up to 3D shapes")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from outer product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()

        if not self.transpose and size_in != size_out:
            raise ValueError("Output shape of broadcast does not match with input shape.")
        if self.transpose and (len(size_out) != 2 or size_out[1] != size_in[0] or size_out[0] != size_in[1]):
            raise ValueError("Output shape of broadcast does not match with input shape.")


##########################################################################################
# End library node
##########################################################################################
