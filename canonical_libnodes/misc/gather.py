# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    This is a dummy library node that acts as a gather to favor canonical 
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
class ExpandGather(ExpandTransformation):
    '''
    Gather that outputs the content of a container by row or columns depending on the row_major properties.
    NOTE: for the moment being this is just a copy operator for the sake of enabling streaming analyzability at the boundaries
    of canonical library nodes (for streaming composition the produced and consumer map must have the same ranges). 
    It would be cool to implement this as a true gather.
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")

        in_desc = in_desc_with_name(node, parent_state, parent_sdfg, "_in")
        out_desc = out_desc_with_name(node, parent_state, parent_sdfg, "_out")
        # # TODO: generalize to multidimension
        N = in_desc.shape[0]
        M = in_desc.shape[1]

        sdfg.add_array("_in", in_desc.shape, in_desc.dtype, strides=in_desc.strides, storage=in_desc.storage)
        sdfg.add_array("_out", out_desc.shape, out_desc.dtype, strides=out_desc.strides, storage=out_desc.storage)

        state = sdfg.add_state(node.label + "_state")
        inp = state.add_read("_in")
        outp = state.add_write("_out")
        # Create map, by row
        gather_tasklet = state.add_tasklet("gather_tasklet", {"_in_val"}, {"_out_val"}, "_out_val = _in_val ")
        if node.row_major:
            gather_map_entry, gather_map_exit = state.add_map("gather_map", {'i': f'0:{N}', 'j': f'0:{M}'})
        else:
            gather_map_entry, gather_map_exit = state.add_map("gather_map", {'j': f'0:{M}', 'i': f'0:{N}'})
        state.add_memlet_path(inp, gather_map_entry, gather_tasklet, dst_conn="_in_val", memlet=dace.Memlet("_in[i,j]"))
        state.add_memlet_path(gather_tasklet,
                              gather_map_exit,
                              outp,
                              src_conn="_out_val",
                              memlet=dace.Memlet("_out[i,j]"))

        # Attempt to use a "tranpose" like impl
        # axes = [1, 0]
        # state.add_mapped_tasklet(
        #     "_transpose_", {f"_i{i}": f"0:{s}"
        #                     for i, s in enumerate(in_desc.shape)},
        #     dict(_in=dace.Memlet(f"_in[{', '.join(f'_i{i}' for i, _ in enumerate(in_desc.shape))}]")),
        #     "_out = _in",
        #     dict(_out=dace.Memlet(f"_out[{', '.join(f'_i{axes[i]}' for i, _ in enumerate(in_desc.shape))}]")),
        #     external_edges=True)
        return sdfg


@dace.library.node
class Gather(dace.sdfg.nodes.LibraryNode):
    '''
        This is a dummy library node that acts as a gather to favor canonical 
        DAG construction and Streaming analysis.
    '''

    # Global properties
    implementations = {
        "gather": ExpandGather,
    }
    default_implementation = None

    # Object fields
    row_major = properties.Property(dtype=bool,
                                    desc="Whether to produce the data in row major (otherwise column major)",
                                    default=True)

    def __init__(self, name, location=None, row_major=True):
        super().__init__(name, location=location, inputs={"_in"}, outputs={"_out"})
        self.row_major = row_major

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

        if len(size_in) > 2:
            raise NotImplementedError("Gather currenly support 2D shapes")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from outer product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()
        if size_out != size_in:
            raise ValueError("Output shape of Gather does not match with input shape.")


##########################################################################################
# End library node
##########################################################################################

# Testing


def create_gather_sdfg(dtype, in_shape, out_shape, row_major, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    inp, inp_arr = sdfg.add_array("inp", in_shape, dtype)
    outp, outp_arr = sdfg.add_array("outp", out_shape, dtype)

    rin = state.add_read("inp")
    wout = state.add_write("outp")
    libnode = Gather('Gather', row_major=row_major)

    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rin, None, libnode, '_in', dace.Memlet.from_array(inp, inp_arr))
    state.add_edge(libnode, '_out', wout, None, dace.Memlet.from_array(outp, outp_arr))

    return sdfg


def run_test(implementation="gather", N=4, M=5):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{N}_{M}"

    # shape of the transposed arrays
    in_shape = [N, M]

    ## Transp not currently supported
    print(f'Gather {N}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    inp = np.random.rand(*in_shape).astype(np_dtype)
    outp = np.random.rand(*in_shape).astype(np_dtype)

    regression = np.copy(inp)

    sdfg = create_gather_sdfg(dace.float32, in_shape, in_shape, True, implementation, sdfg_name)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    sdfg(inp=inp, outp=outp)

    assert np.allclose(outp, regression)


if __name__ == "__main__":
    run_test()
