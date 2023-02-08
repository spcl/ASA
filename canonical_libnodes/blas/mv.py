# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical LibNode and expansion for Matrix-Vector multiplication computing y=AX.
    The expansion is compatible with DaCe GEMV library node.
"""

import dace
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from dace import properties
from copy import deepcopy as dc

#######################################################################
# MV Library Node
#######################################################################


@dace.library.expansion
class ExpandMVPure(ExpandTransformation):

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
            raise NotImplementedError("Vectorization for pure GEMV NYI.")

        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if trans_shape_a[1] != shape_x[0]:
            raise SyntaxError("Matrix-vector product size mismatch: {} vs. {}".format(trans_shape_a[1], shape_x[0]))

        N, M = trans_shape_a[0], trans_shape_a[1]

        if outer_array_a.storage != outer_array_x.storage:
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_A", shape_a, dtype_a, strides=strides_a, storage=storage)
        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, strides=strides_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, strides=strides_y, storage=storage)

        mul_program = "__out = {} * __A * __x".format(node.alpha)

        init_state = sdfg.add_state(node.label + "_initstate")
        state = sdfg.add_state_after(init_state, node.label + "_state")

        if node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(shape_y, dtype_y, storage=storage)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        # Initialization map
        init_state.add_mapped_tasklet(
            "gemv_init", {"_o%d" % i: "0:%s" % symstr(d)
                          for i, d in enumerate(shape_y)}, {},
            "out = 0",
            {"out": dace.Memlet("{}[{}]".format(mul_out, ",".join(["_o%d" % i for i in range(len(shape_y))])))},
            external_edges=True)

        # Multiplication map
        state.add_mapped_tasklet("_GEMV_", {"__i%d" % i: "0:%s" % s
                                            for i, s in enumerate([N, M])},
                                 {
                                     "__A": dace.Memlet("_A[{}]".format("__i1, __i0" if node.transA else "__i0, __i1")),
                                     "__x": dace.Memlet("_x[__i1]")
                                 },
                                 mul_program, {"__out": dace.Memlet(f"{mul_out}[__i0]", wcr="lambda x, y: x + y")},
                                 external_edges=True,
                                 output_nodes=output_nodes)

        add_program = "__y_out = ({} * __y_in) + __tmp".format(node.beta)

        memlet_idx = "__i"

        # addition map
        if node.beta != 0:
            state.add_mapped_tasklet("_Add_", {"__i": "0:{}".format(N)}, {
                "__y_in": dace.Memlet(f"_y[{memlet_idx}]"),
                "__tmp": dace.Memlet(f"{mul_out}[__i]"),
            },
                                     add_program, {"__y_out": dace.Memlet("_y[__i]")},
                                     external_edges=True,
                                     input_nodes={mul_out: access_tmp})
        return sdfg


@dace.library.expansion
class ExpandMVSeq(ExpandTransformation):
    '''
    Sequential MV implementation that does not require init state
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
            raise NotImplementedError("Vectorization for pure MV NYI.")

        if node.transA:
            raise RuntimeError("Currently not supported")
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        # A dimensions may be squeezed out
        if len(trans_shape_a) == 1:
            N = 1
            M = trans_shape_a[0]
            strides_y = [1]  # this is a dot product
            shape_a = [N, M]
            if len(strides_a) == 0:
                strides_a = [M, 1]
            else:
                strides_a = [M, strides_a[-1]]
        else:
            N, M = trans_shape_a[0], trans_shape_a[1]

        if M != shape_x[-1]:
            import pdb
            pdb.set_trace()
            raise SyntaxError("Matrix-vector product size mismatch: {} vs. {}".format(M, shape_x[0]))

        if not outer_array_a.transient and not outer_array_x.transient and outer_array_a.storage != outer_array_x.storage:
            # DaCe automatically set the storage type for transients to register
            # Skip this check in that case
            raise ValueError("Input matrices must have same storage")
        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_A", shape_a, dtype_a, strides=strides_a, storage=storage)
        _, array_x = sdfg.add_array("_x", shape_x, dtype_x, strides=strides_x, storage=storage)
        _, array_y = sdfg.add_array("_y", shape_y, dtype_y, strides=strides_y, storage=storage)

        state = sdfg.add_state(node.label + "_state")

        if node.beta == 0:
            mul_out, mul_out_array = "_y", array_y
            output_nodes = None
        else:
            mul_out, mul_out_array = tmp, array_tmp = sdfg.add_temp_transient(shape_y, dtype_y, storage=storage)

            access_tmp = state.add_read(tmp)
            output_nodes = {mul_out: access_tmp}

        is_batched = node.used_in_batched_matmul if hasattr(
            node, 'used_in_batched_matmul') else False  # Used only with ONNX Matmul

        if is_batched:
            BS = shape_x[0]
            batch_entry, batch_exit = state.add_map("batch", {"b": f"0:{BS}"}, schedule=dace.ScheduleType.Sequential)
            row_entry, row_exit = state.add_map("rows", {"i": f"0:{N//BS}"}, schedule=dace.ScheduleType.Sequential)
        else:
            # Create rows map
            row_entry, row_exit = state.add_map("rows", {"i": f"0:{N}"}, schedule=dace.ScheduleType.Sequential)

        # create cols map
        col_entry, col_exit = state.add_map("cols", {"j": f"0:{M}"}, schedule=dace.ScheduleType.Sequential)

        # Local buffer of x
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

        state.add_memlet_path(row_entry, init_accum, memlet=dace.Memlet())
        state.add_memlet_path(init_accum, accum_access_in, src_conn="acc_out", memlet=dace.Memlet(f"accum[0]"))

        if is_batched:
            # Multiplication
            state.add_memlet_path(read_a,
                                  batch_entry,
                                  row_entry,
                                  col_entry,
                                  multiply_tasklet,
                                  dst_conn="A_in",
                                  memlet=dace.Memlet(f"_A[b*{N//BS} + i,j]"))

            state.add_memlet_path(read_x,
                                  batch_entry,
                                  row_entry,
                                  col_entry,
                                  multiply_tasklet,
                                  dst_conn="x_in",
                                  memlet=dace.Memlet(f"_x[b, j]"))

        else:
            # Multiplication
            state.add_memlet_path(read_a,
                                  row_entry,
                                  col_entry,
                                  multiply_tasklet,
                                  dst_conn="A_in",
                                  memlet=dace.Memlet(f"_A[i,j]"))

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
        # if this is degenerate MV (dot product), write directly to 0 to help SDFG inlining
        if not is_batched:
            if N == 1:
                state.add_memlet_path(accum_access_out, row_exit, write_y, memlet=dace.Memlet(f"_y[0]"))
            else:
                state.add_memlet_path(accum_access_out, row_exit, write_y, memlet=dace.Memlet(f"_y[i]"))
        else:
            state.add_memlet_path(accum_access_out,
                                  row_exit,
                                  batch_exit,
                                  write_y,
                                  memlet=dace.Memlet(f"_y[b*{N//BS} + i]"))
        return sdfg


@dace.library.node
class MV(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        #"pure": ExpandMVPure,
        "seq": ExpandMVSeq,
    }
    default_implementation = "seq"

    # Object fields
    alpha = properties.SymbolicProperty(allow_none=False, default=1)
    beta = properties.SymbolicProperty(allow_none=False, default=0)

    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")

    n = properties.SymbolicProperty(allow_none=True, default=None)
    m = properties.SymbolicProperty(allow_none=True, default=None)

    used_in_batched_matmul = properties.Property(dtype=bool,
                                                 desc="Whether this is used in a Batched matmul (ONNX node only)")

    def __init__(self, name, location=None, transA=False, alpha=1, beta=0, used_in_batched_matmul=False):
        super().__init__(name,
                         location=location,
                         inputs={"_A", "_x", "_y"} if beta != 0 else {"_A", "_x"},
                         outputs={"_y"})
        self.transA = transA
        self.alpha = alpha
        self.beta = beta
        self.used_in_batched_matmul = used_in_batched_matmul

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2, 3]:
            raise ValueError("Expected 2 or 3 inputs to GEMV")
        size_y_in = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == "_A":
                subset_a = dc(memlet.subset)
                # subset_a.squeeze()
                # Note: in case of degenerated matrix (1xM size), the squeezing sees only one dimension
                size_a = subset_a.size()
            if dst_conn == "_x":
                subset = dc(memlet.subset)
                subset.squeeze()
                size_x = subset.size()
            if dst_conn == "_y":
                subset = dc(memlet.subset)
                subset.squeeze()
                size_y_in = subset.size()

        if len(size_a) != 2 or (self.used_in_batched_matmul and len(size_x) != 2) or (not self.used_in_batched_matmul
                                                                                      and len(size_x) != 1):

            import pdb
            pdb.set_trace()
            raise ValueError("Matrix-vector product only supported on matrix-vector input")

        a_cols = size_a[1] if not self.transA else size_a[0]
        a_rows = size_a[0] if not self.transA else size_a[1]

        if a_cols != size_x[-1]:
            raise ValueError(f"Columns of A ({a_cols}) don't match " f"size of x ({size_x[0]}).")

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-vector product")
        out_memlet = out_edges[0].data

        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size_y_out = out_subset.size()
        if size_y_in is not None and size_y_in != size_y_out:
            raise ValueError("Input y-vector must match output y-vector.")
        if (len(size_y_out) != 1 or size_y_out[0] != a_rows):
            raise ValueError("Vector input to GEMV must match matrix rows.")


#####################################################################################
# END LIBNODE
#####################################################################################