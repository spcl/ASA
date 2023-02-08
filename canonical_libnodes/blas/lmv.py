# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical expansion for a  a Left- Matrix Vector multiplication
    This computes y = xA, where x is a row vector of K elements, A is a matrix
    KxM and the result will be a row vector of size M
"""

import dace
import random
import numpy as np
from dace import dtypes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from dace import properties
from copy import deepcopy as dc
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG
from dace.transformation.dataflow import StreamingComposition
from dace.memlet import Memlet

#######################################################################
# LMV Library Node
#######################################################################


@dace.library.expansion
class ExpandLMVCol(ExpandTransformation):
    '''
    Sequential LMV implementation: x is multiplied with a column of A to compute
    one element of y
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        ((edge_a, outer_array_a, shape_a, strides_a), (edge_x, outer_array_x, shape_x, strides_x),
         (edge_y, outer_array_y, shape_y, strides_y)) = _get_matmul_operands(node,
                                                                             parent_state,
                                                                             parent_sdfg,
                                                                             name_lhs="_A",
                                                                             name_rhs="_x",
                                                                             name_out="_y")
        dtype_a = outer_array_a.dtype.type
        dtype_x = outer_array_x.dtype.type
        dtype_y = outer_array_y.dtype.type

        if outer_array_a.dtype.veclen > 1 or outer_array_x.dtype.veclen > 1:
            raise NotImplementedError("Vectorization for LMV NYI.")

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.beta != 0:
            raise NotImplementedError("Beta different than zero is not currently supported")
        if trans_shape_a[-2] != shape_x[-1]:
            raise SyntaxError("Left Matrix-vector product size mismatch: {} vs. {}".format(
                trans_shape_a[-2], shape_x[0]))

        K, M = trans_shape_a[-2], trans_shape_a[-1]

        if not outer_array_a.transient and not outer_array_x.transient and outer_array_a.storage != outer_array_x.storage:
            # DaCe automatically set the storage type for transients to register
            # Skip this check in that case
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_A", shape_a, dtype_a, strides=strides_a, storage=storage)
        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, strides=strides_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, strides=strides_y, storage=storage)

        state = sdfg.add_state(node.label + "_state")

        if not node.transA:
            # TODO: complete this one, not sure how easy it is
            if node.used_in_batched_matmul:
                BS = shape_a[0]  # TODO: what if BS=1?
                N = shape_x[0] // BS
                batch_entry, batch_exit = state.add_map("batch", {"b": f"0:{BS}"},
                                                        schedule=dace.ScheduleType.Sequential)
            # row_entry, row_exit = state.add_map("rows", {"i": f"0:{N//BS}"}, schedule=dace.ScheduleType.Sequential)

            # Create y-elements map
            col_entry, col_exit = state.add_map("cols", {"j": f"0:{M}"}, schedule=dace.ScheduleType.Sequential)

            # create map for rows of A, elements of x
            row_entry, row_exit = state.add_map("rows", {"k": f"0:{K}"}, schedule=dace.ScheduleType.Sequential)
        else:
            assert not node.used_in_batched_matmul
            # Create y-elements map
            col_entry, col_exit = state.add_map("cols", {"j": f"0:{K}"}, schedule=dace.ScheduleType.Sequential)

            # create map for rows of A, elements of x
            row_entry, row_exit = state.add_map("rows", {"k": f"0:{M}"}, schedule=dace.ScheduleType.Sequential)

        # Local buffer for accumulation
        sdfg.add_scalar("accum", dtype_y, storage=dace.StorageType.Register, transient=True)
        accum_access_in = state.add_access("accum")
        accum_access_out = state.add_access("accum")

        read_a = state.add_read("_A")
        read_x = state.add_read("_x")
        write_y = state.add_read("_y")

        # Multiply tasklet
        multiply_tasklet = state.add_tasklet("multiply", {"A_in", "x_in", "accum_in"}, {"accum_out"},
                                             "accum_out = A_in * x_in + accum_in")

        # TODO: init accum
        init_accum = state.add_tasklet("init_accum", {}, {"acc_out"}, "acc_out = 0")

        if not node.transA:

            state.add_memlet_path(col_entry, init_accum, memlet=dace.Memlet())
            state.add_memlet_path(init_accum, accum_access_in, src_conn="acc_out", memlet=dace.Memlet(f"accum[0]"))

            # Multiplication, accumulate to compute y[j]
            if not node.used_in_batched_matmul:
                state.add_memlet_path(read_a,
                                      col_entry,
                                      row_entry,
                                      multiply_tasklet,
                                      dst_conn="A_in",
                                      memlet=dace.Memlet(f"_A[k,j]"))

                state.add_memlet_path(read_x,
                                      col_entry,
                                      row_entry,
                                      multiply_tasklet,
                                      dst_conn="x_in",
                                      memlet=dace.Memlet(f"_x[k]"))
            else:
                state.add_memlet_path(read_a,
                                      batch_entry,
                                      col_entry,
                                      row_entry,
                                      multiply_tasklet,
                                      dst_conn="A_in",
                                      memlet=dace.Memlet(f"_A[b, k,j]"))

                state.add_memlet_path(read_x,
                                      batch_entry,
                                      col_entry,
                                      row_entry,
                                      multiply_tasklet,
                                      dst_conn="x_in",
                                      memlet=dace.Memlet(f"_x[b*{N}, k]"))

            state.add_memlet_path(accum_access_in,
                                  row_entry,
                                  multiply_tasklet,
                                  dst_conn="accum_in",
                                  memlet=dace.Memlet(f"accum[0]"))
            state.add_memlet_path(multiply_tasklet,
                                  row_exit,
                                  accum_access_out,
                                  src_conn="accum_out",
                                  memlet=dace.Memlet(f"accum[0]"))
            # Write the result out
            if not node.used_in_batched_matmul:
                state.add_memlet_path(accum_access_out, col_exit, write_y, memlet=dace.Memlet(f"_y[j]"))
            else:
                state.add_memlet_path(accum_access_out,
                                      col_exit,
                                      batch_exit,
                                      write_y,
                                      memlet=dace.Memlet(f"_y[b*{N}+{node.id_in_batched_matmul}, j]"))

        else:
            assert not node.used_in_batched_matmul
            # in this case we should multiply x with a row of A to produce one element of y
            state.add_memlet_path(row_entry, init_accum, memlet=dace.Memlet())
            state.add_memlet_path(init_accum, accum_access_in, src_conn="acc_out", memlet=dace.Memlet(f"accum[0]"))

            # Multiplication, accumulate to compute y[j]
            state.add_memlet_path(read_a,
                                  row_entry,
                                  col_entry,
                                  multiply_tasklet,
                                  dst_conn="A_in",
                                  memlet=dace.Memlet(f"_A[k,j]"))

            state.add_memlet_path(read_x,
                                  row_entry,
                                  col_entry,
                                  multiply_tasklet,
                                  dst_conn="x_in",
                                  memlet=dace.Memlet(f"_x[j]"))
            state.add_memlet_path(accum_access_in,
                                  col_entry,
                                  multiply_tasklet,
                                  dst_conn="accum_in",
                                  memlet=dace.Memlet(f"accum[0]"))
            state.add_memlet_path(multiply_tasklet,
                                  col_exit,
                                  accum_access_out,
                                  src_conn="accum_out",
                                  memlet=dace.Memlet(f"accum[0]"))
            # Write the result out
            state.add_memlet_path(accum_access_out, row_exit, write_y, memlet=dace.Memlet(f"_y[k]"))
        # sdfg.view()
        return sdfg


@dace.library.node
class LMV(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "LMV_col": ExpandLMVCol,
    }
    default_implementation = "LMV_col"

    # Object fields
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")

    n = properties.SymbolicProperty(allow_none=True, default=None)
    m = properties.SymbolicProperty(allow_none=True, default=None)

    used_in_batched_matmul = properties.Property(dtype=bool,
                                                 desc="Whether this is used in a Batched matmul (ONNX node only)")

    id_in_batched_matmul = properties.Property(dtype=int, desc="ID when used in a Batched matmul (ONNX node only)")

    def __init__(self, name, location=None, transA=False, alpha=1, beta=0, used_in_batched_matmul=False, id=0):
        super().__init__(name,
                         location=location,
                         inputs={"_A", "_x", "_y"} if beta != 0 else {"_A", "_x"},
                         outputs={"_y"})
        self.transA = transA
        self.alpha = alpha
        self.beta = beta
        self.used_in_batched_matmul = used_in_batched_matmul
        self.id_in_batched_matmul = id

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to GEMV")
        size_y_in = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                subset = dc(memlet.subset)
                subset.squeeze()
                size_a = subset.size()
            if dst_conn == "_x":
                subset = dc(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
            if dst_conn == "_y":
                subset = dc(memlet.subset)
                subset.squeeze()
                size_y_in = subset.size()

        if not self.used_in_batched_matmul:
            if len(size_a) != 2 or len(size_x) != 1:

                raise ValueError("Matrix-vector product only supported on matrix-vector input")
        else:
            if len(size_a) != 3 or len(size_x) > 2:
                raise ValueError("Matrix-vector product only supported on matrix-vector input")

        a_cols = size_a[-1] if not self.transA else size_a[-2]
        a_rows = size_a[-2] if not self.transA else size_a[-1]

        if a_rows != size_x[-1]:
            raise ValueError(f"Rows of A ({a_rows}) don't match " f"size of x ({size_x[0]}).")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-vector product")
        out_memlet = out_edges[0].data

        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size_y_out = out_subset.size()
        if size_y_in is not None and size_y_in != size_y_out:
            raise ValueError("Input y-vector must match output y-vector.")
        if not self.used_in_batched_matmul:
            if (len(size_y_out) != 1 or size_y_out[0] != a_cols):
                raise ValueError("Vector input to LMV must match matrix cols.")
        else:
            if (len(size_y_out) != 2 or size_y_out[-1] != a_cols):
                import pdb
                pdb.set_trace()
                raise ValueError("Vector input to LMV must match matrix cols.")


#####################################################################################
# END LIBNODE
#####################################################################################
