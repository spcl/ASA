# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    This is a library node that acts as a reducer along a given axis
    
    TODO: generalize

'''
import dace
import numpy as np
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
import copy
from dace import library
from dace import properties
from dace.transformation.interstate import InlineSDFG

N, M = (dace.symbol(s, dtype=dace.int32) for s in ('N', 'M'))


@dace.library.expansion
class ExpandReductionCol(ExpandTransformation):
    '''
    Reduction by column
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")

        in_desc = in_desc_with_name(node, parent_state, parent_sdfg, "_in")
        out_desc = out_desc_with_name(node, parent_state, parent_sdfg, "_out")
        # # TODO: generalize to multidimension
        K = in_desc.shape[0]
        N = in_desc.shape[1]
        M = in_desc.shape[2]

        sdfg.add_array("_in", in_desc.shape, in_desc.dtype, strides=in_desc.strides, storage=in_desc.storage)
        sdfg.add_array("_out", out_desc.shape, out_desc.dtype, strides=out_desc.strides, storage=out_desc.storage)

        state = sdfg.add_state(node.label + "_state")
        inp = state.add_read("_in")
        outp = state.add_write("_out")

        sdfg.add_scalar("accum", out_desc.dtype, storage=dace.StorageType.Register, transient=True)
        accum_access_in = state.add_access("accum")
        accum_access_out = state.add_access("accum")
        init_accum = state.add_tasklet("init_accum", {}, {"acc_out"}, "acc_out = 0")

        # Note: we proceed col by col
        out_map_entry, out_map_exit = state.add_map("out_map", {
            "j": f"0:{M}",
            "i": f"0:{N}"
        },
                                                    schedule=dace.ScheduleType.Default)
        sum_map_entry, sum_map_exit = state.add_map("sum_map", {
            "k": f"0:{K}",
        }, schedule=dace.ScheduleType.Sequential)

        sum_out = state.add_tasklet("sum", {"_in_value", "_accum"}, {"_out"}, "_out = _in_value + _accum ")

        # Init accum
        state.add_memlet_path(out_map_entry, init_accum, memlet=dace.Memlet())
        state.add_memlet_path(init_accum, accum_access_in, src_conn="acc_out", memlet=dace.Memlet(f"accum[0]"))

        # Accum -> tasklet
        state.add_memlet_path(accum_access_in,
                              sum_map_entry,
                              sum_out,
                              dst_conn="_accum",
                              memlet=dace.Memlet(f"accum[0]"))

        # partial results -> sum tasklet
        state.add_memlet_path(inp,
                              out_map_entry,
                              sum_map_entry,
                              sum_out,
                              dst_conn="_in_value",
                              memlet=dace.Memlet(f"_in[k, i, j]"))
        # save accum_result
        state.add_memlet_path(sum_out, sum_map_exit, accum_access_out, src_conn="_out", memlet=dace.Memlet(f"accum[0]"))

        # then write to final C
        state.add_memlet_path(accum_access_out, out_map_exit, outp, memlet=dace.Memlet("_out[i,j]"))

        return sdfg


@dace.library.node
class ReduceMMM(dace.sdfg.nodes.LibraryNode):
    '''
        Reduce library node for MMM
    '''

    # Global properties
    implementations = {
        "reduce_sum_col": ExpandReductionCol,
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
            raise ValueError("Expected 1 inputs to Reduce")

        in_memlet = in_edges[0].data

        in_subset = copy.deepcopy(in_memlet.subset)
        in_subset.squeeze()
        size_in = in_subset.size()

        if len(size_in) > 3:  # it could happens that one of dimension is squeezed
            raise NotImplementedError("Reduce currenly support 3D shapes")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from reduce")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()
        if (len(size_out) == 2 and
            (size_out[0] != size_in[1] or size_out[1] != size_in[2])) or (len(size_out) == 1 and
                                                                          (size_out[0] != size_in[1])):
            raise ValueError("Output shape of Reduce does not match with input shape.")


##########################################################################################
# End library node
##########################################################################################
