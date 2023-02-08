# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical Library Node and expansion for Outer-Product between two vectors u and v
"""
import dace
import numpy as np
from dace.transformation.transformation import ExpandTransformation
import copy
from dace import properties
from dace.libraries.blas.nodes.matmul import _get_matmul_operands


@dace.library.expansion
class ExpandOPColumn(ExpandTransformation):
    '''
    OP implementation that outputs the result by column: the entire u is used to multiply
    with one element of v, producing a column of the final result
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")

        # get the ACTUAL shape/strides of incoming data (note, if we only pick the descriptor we
        # don't see the actual size of data arriving)
        ((edge_u, outer_array_u, shape_u, strides_u), (edge_v, outer_array_v, shape_v, strides_v),
         (edge_A, outer_array_A, shape_A, strides_A)) = _get_matmul_operands(node,
                                                                             parent_state,
                                                                             parent_sdfg,
                                                                             name_lhs="_u",
                                                                             name_rhs="_v",
                                                                             name_out="_A")
        # TODO: checks
        N = shape_u[-1]
        M = shape_v[-1]

        if N == 1:
            strides_u = [1]
            shape_A = [N, M]
            strides_A = [M, strides_A[-1]]
        if M == 1:
            strides_v = [1]
            shape_A = [N, M]
            strides_A = [1, 1]  # TODO: not sure

        dtype_u = outer_array_u.dtype.type
        dtype_v = outer_array_v.dtype.type
        dtype_A = outer_array_A.dtype.type

        sdfg.add_array("_u", shape_u, dtype_u, strides=strides_u, storage=outer_array_u.storage)
        sdfg.add_array("_v", shape_v, dtype_v, strides=strides_v, storage=outer_array_v.storage)
        sdfg.add_array("_A", shape_A, dtype_A, strides=strides_A, storage=outer_array_A.storage)

        state = sdfg.add_state(node.label + "_state")

        if node.used_in_batched_matmul:
            # TODO: this version has not been tested!!!
            BS = shape_v[0]
            batch_entry, batch_exit = state.add_map("batch", {"b": f"0:{BS}"}, schedule=dace.ScheduleType.Sequential)
            # This is used only with ONNX MatMul node and its canonical expansion. A is flattened in 2D matrix, having
            # size BS*N x K
            map_entry, map_exit = state.add_map("op_map", {
                "j": f"0:{M}",
                "i": f"0:{N//BS}"
            },
                                                schedule=dace.ScheduleType.Default)

        else:

            # Create map
            map_entry, map_exit = state.add_map("op_map", {
                "j": f"0:{M}",
                "i": f"0:{N}"
            },
                                                schedule=dace.ScheduleType.Default)

        read_u = state.add_read("_u")
        read_v = state.add_read("_v")
        write_A = state.add_read("_A")

        # Multiply tasklet
        multiply_tasklet = state.add_tasklet("multiply", {"u_in", "v_in"}, {"A_out"}, "A_out = u_in * v_in")

        # Multiplication
        if node.used_in_batched_matmul:
            state.add_memlet_path(read_u,
                                  batch_entry,
                                  map_entry,
                                  multiply_tasklet,
                                  dst_conn="u_in",
                                  memlet=dace.Memlet(f"_u[b*{N//BS}+i]"))

            state.add_memlet_path(read_v,
                                  batch_entry,
                                  map_entry,
                                  multiply_tasklet,
                                  dst_conn="v_in",
                                  memlet=dace.Memlet(f"_v[b, j]"))
            state.add_memlet_path(multiply_tasklet,
                                  map_exit,
                                  batch_exit,
                                  write_A,
                                  src_conn="A_out",
                                  memlet=dace.Memlet(f"_A[b*{N//BS}+i,j]"))
        else:
            state.add_memlet_path(read_u, map_entry, multiply_tasklet, dst_conn="u_in", memlet=dace.Memlet(f"_u[i]"))

            state.add_memlet_path(read_v, map_entry, multiply_tasklet, dst_conn="v_in", memlet=dace.Memlet(f"_v[j]"))
            state.add_memlet_path(multiply_tasklet, map_exit, write_A, src_conn="A_out", memlet=dace.Memlet(f"_A[i,j]"))

        return sdfg


class ExpandOPRow(ExpandTransformation):
    '''
    OP implementation that outputs the result by row: one element of u is used to multiply
    the entire v, producing one row of the final result
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")

        # get the ACTUAL shape/strides of incoming data (note, if we only pick the descriptor we
        # don't see the actual size of data arriving)
        ((edge_u, outer_array_u, shape_u, strides_u), (edge_v, outer_array_v, shape_v, strides_v),
         (edge_A, outer_array_A, shape_A, strides_A)) = _get_matmul_operands(node,
                                                                             parent_state,
                                                                             parent_sdfg,
                                                                             name_lhs="_u",
                                                                             name_rhs="_v",
                                                                             name_out="_A")
        # TODO: checks
        N = shape_u[-1]
        M = shape_v[-1]

        if N == 1:
            strides_u = [1]
            shape_A = [N, M]
            strides_A = [M, strides_A[-1]]
        if M == 1:
            strides_v = [1]
            shape_A = [N, M]
            strides_A = [1, 1]  # TODO: not sure

        dtype_u = outer_array_u.dtype.type
        dtype_v = outer_array_v.dtype.type
        dtype_A = outer_array_A.dtype.type

        sdfg.add_array("_u", shape_u, dtype_u, strides=strides_u, storage=outer_array_u.storage)
        sdfg.add_array("_v", shape_v, dtype_v, strides=strides_v, storage=outer_array_v.storage)
        sdfg.add_array("_A", shape_A, dtype_A, strides=strides_A, storage=outer_array_A.storage)

        state = sdfg.add_state(node.label + "_state")

        # Create map

        if node.used_in_batched_matmul:
            # TODO: this version has not been tested!!!
            BS = shape_v[0]
            batch_entry, batch_exit = state.add_map("batch", {"b": f"0:{BS}"}, schedule=dace.ScheduleType.Sequential)
            # This is used only with ONNX MatMul node and its canonical expansion. A is flattened in 2D matrix, having
            # size BS*N x K
            map_entry, map_exit = state.add_map("op_map", {
                "i": f"0:{N//BS}",
                "j": f"0:{M}"
            },
                                                schedule=dace.ScheduleType.Default)

        else:
            map_entry, map_exit = state.add_map("op_map", {
                "i": f"0:{N}",
                "j": f"0:{M}"
            },
                                                schedule=dace.ScheduleType.Default)

        read_u = state.add_read("_u")
        read_v = state.add_read("_v")
        write_A = state.add_read("_A")

        # Multiply tasklet
        multiply_tasklet = state.add_tasklet("multiply", {"u_in", "v_in"}, {"A_out"}, "A_out = u_in * v_in")

        # Multiplication
        if node.used_in_batched_matmul:
            state.add_memlet_path(read_u,
                                  batch_entry,
                                  map_entry,
                                  multiply_tasklet,
                                  dst_conn="u_in",
                                  memlet=dace.Memlet(f"_u[b*{N//BS}+i]"))
            state.add_memlet_path(read_v,
                                  batch_entry,
                                  map_entry,
                                  multiply_tasklet,
                                  dst_conn="v_in",
                                  memlet=dace.Memlet(f"_v[b, j]"))
            state.add_memlet_path(multiply_tasklet,
                                  map_exit,
                                  batch_exit,
                                  write_A,
                                  src_conn="A_out",
                                  memlet=dace.Memlet(f"_A[b*{N//BS}+i,j]"))

        else:
            state.add_memlet_path(read_u, map_entry, multiply_tasklet, dst_conn="u_in", memlet=dace.Memlet(f"_u[i]"))
            state.add_memlet_path(read_v, map_entry, multiply_tasklet, dst_conn="v_in", memlet=dace.Memlet(f"_v[j]"))
            state.add_memlet_path(multiply_tasklet, map_exit, write_A, src_conn="A_out", memlet=dace.Memlet(f"_A[i,j]"))
        # sdfg.view()

        return sdfg


@dace.library.node
class OP(dace.sdfg.nodes.LibraryNode):
    '''
        Computes A = alpha * u * v.T
        where u is a vector of size N and v is a vector of size M
        A will be of size NxM
    '''

    # Global properties
    implementations = {"OP_by_col": ExpandOPColumn, "OP_by_row": ExpandOPRow}
    default_implementation = "OP_by_col"

    # Object fields
    alpha = properties.SymbolicProperty(allow_none=False, default=1)

    # transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")

    n = properties.SymbolicProperty(allow_none=True, default=None, desc="Size of input vector u")
    m = properties.SymbolicProperty(allow_none=True, default=None, desc="Size of input vector v")

    used_in_batched_matmul = properties.Property(dtype=bool,
                                                 desc="Whether this is used in a Batched matmul (ONNX node only)")

    def __init__(self, name, location=None, transA=False, alpha=1, beta=0, used_in_batched_matmul=False):
        super().__init__(name, location=location, inputs={"_u", "_v"}, outputs={"_A"})
        self.alpha = alpha
        self.used_in_batched_matmul = used_in_batched_matmul

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2]:
            raise ValueError("Expected 2 inputs to OP")
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_u":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_u = subset.size()
            if dst_conn == "_v":
                subset = copy.deepcopy(memlet.subset)
                subset.squeeze()
                size_v = subset.size()

        if not self.used_in_batched_matmul and (len(size_u) != 1 or len(size_v) != 1):
            raise ValueError("Outer product supported only for vectors")

        if self.alpha != 1:
            raise NotImplementedError("Alpha must be 1")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from outer product")
        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_A_out = out_subset.size()
        if not self.used_in_batched_matmul:
            if (len(size_A_out) == 2 and
                (size_A_out[0] != size_u[0] or size_A_out[1] != size_v[0])) or (len(size_A_out) == 1 and size_u[0] == 1
                                                                                and size_A_out[0] != size_v[0]):
                raise ValueError("Output shape of OP does not match with input shapes.")
        else:
            if size_A_out[0] != size_u[-1] or size_A_out[1] != size_v[-1]:
                raise ValueError("Output shape of OP does not match with input shapes.")


##########################################################################################
# End library node
##########################################################################################
