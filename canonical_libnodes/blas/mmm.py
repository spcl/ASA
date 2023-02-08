# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical LibNode and expansions for Matrix-Matrix Multiplication (MMM).

    Expansions are compatible with DaCe GEMM library node.
"""

# TODO:
# - support accumulation on resulting matrix (beta != 0)
# - add more expansions

import dace
import numpy as np
from typing import Set
from dace import dtypes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.blas.nodes.matmul import (_get_matmul_operands, _get_codegen_gemm_opts)
from dace import properties
from copy import deepcopy as dc
from dace.memlet import Memlet
from canonical_libnodes.blas.mv import MV
from canonical_libnodes.blas.op import OP
from canonical_libnodes.blas.lmv import LMV
from canonical_libnodes.misc.gather import Gather
from canonical_libnodes.misc.broadcast import Broadcast
from canonical_libnodes.misc.reduction import ReduceMMM


@dace.library.expansion
class ExpandMMM_MV(ExpandTransformation):
    '''
    In this expansion we have as many MV as M (the number of columns of matrix C).
    Each one of them will take in input the entire A and a column of B.
    The final matrix C will be produces row-by-row
    '''

    environments = []
    supported_flags: Set[str] = {'mv'}

    @staticmethod
    def can_be_used(flags: Set[str]):
        """
        Returns whether this expansion supports at least one of the flags provided as arguments.
        """
        return bool(flags & ExpandMMM_MV.supported_flags)

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b, shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_b).type]

        is_batched = node.batched if hasattr(node, 'batched') else False  # Used only with ONNX Matmul

        ### Transposition currently not supported
        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        # if (len(trans_shape_a) != 2 or len(trans_shape_b) != 2 or trans_shape_a[1] != trans_shape_b[0]):
        #     raise SyntaxError("Matrix sizes must match")
        # one dimension could be one
        # Deal with degenerated cases (N or M equal to 1)
        if len(trans_shape_a) == 1:
            N = 1
            K = trans_shape_a[0]

            shape_a = (N, K)
            if len(strides_a) == 0:
                strides_a = [K, 1]
            else:
                strides_a = [K, strides_a[-1]]
        else:
            N = trans_shape_a[-2]
            K = trans_shape_a[-1]

        if len(trans_shape_b) == 1:
            M = 1
            if node.transB:
                shape_b = (M, K)
                if len(strides_b) == 0:
                    strides_b = [K, 1]
                else:
                    strides_b = [K, strides_a[-1]]

            else:
                shape_b = (K, M)
                if len(strides_b) == 0:
                    strides_b = [M, 1]
                else:
                    strides_b = [M, strides_a[-1]]

        else:
            M = trans_shape_b[-1]

        if is_batched:
            # In this case there is another dimension
            BS = trans_shape_b[0]
        else:
            BS = 1
        shape_c = (N, M)

        if N == 1 or M == 1:
            stride_c = [M, *cdata[-1]]
        else:
            stride_c = cdata[-1]

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=stride_c, storage=cdata[1].storage)

        state = sdfg.add_state(node.label + "_state")

        # In this case B must be always in a buffer node. We add it explicitly to facilitate canonical DAG construction
        _, array_b_bn = sdfg.add_array("_b_bn",
                                       shape_b,
                                       dtype_b,
                                       strides=strides_b,
                                       storage=outer_array_b.storage,
                                       transient=True)

        B_in = state.add_read("_b")
        B = state.add_access("_b_bn")
        state.add_edge(B_in, None, B, None, dace.Memlet.from_array('_b', array_b))

        if node.alpha == 1.0:
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        ## Computation State
        A = state.add_read("_a")

        # Broadcast (fake broadcast) node that will broadcast A to all the MV nodes (see below).
        # This is currently implemented as a copy to a transient, and we do this to favor streaming composition (we are interested)
        # in the access patter to A, that in this case is by row
        _, array_broadc_A = sdfg.add_array("_broadcasted_A",
                                           shape_a,
                                           dtype_a,
                                           transient=True,
                                           strides=strides_a,
                                           storage=outer_array_a.storage)

        broadcast_A = state.add_access("_broadcasted_A")
        broadcast_libnode = Broadcast('Broadcast_A_MMM', row_major=True)

        broadcast_libnode.implementation = "broadcast"
        state.add_node(broadcast_libnode)

        state.add_edge(A, None, broadcast_libnode, '_in', dace.Memlet.from_array('_a', array_a))
        state.add_edge(broadcast_libnode, '_out', broadcast_A, None,
                       dace.Memlet.from_array('_broadcasted_A', array_broadc_A))

        # To favor streaming composition, we need to add a transient, that is used to combine things, and then we copy
        # to the final container using the exact pattern that would be used in the actual computation
        # In this case, the final matrix C will be produced row by row
        _, array_tmp = sdfg.add_array("_tmp",
                                      shape_c,
                                      dtype_c,
                                      transient=True,
                                      strides=stride_c,
                                      storage=cdata[1].storage)
        tmp = state.add_access("_tmp")
        C = state.add_write("_c")

        # Add the MV library nodes
        # from dace.libraries.blas import Gemv
        mv_implementation = "seq"
        # Note: memlet to MV already account for the right data volume

        for i in range(M):
            gemv_node = MV("mv", alpha=node.alpha, beta=0, used_in_batched_matmul=is_batched)
            gemv_node.implementation = mv_implementation
            state.add_memlet_path(broadcast_A, gemv_node, dst_conn="_A", memlet=Memlet(f"_broadcasted_A[0:{N}, 0:{K}]"))
            if node.transB:
                # one row of B must be sent to the MV
                state.add_memlet_path(B, gemv_node, dst_conn="_x", memlet=Memlet(f"_b_bn[{i}, 0:{K}]"))
            else:
                # one column must be sent to the MV
                if not is_batched:
                    state.add_memlet_path(B, gemv_node, dst_conn="_x", memlet=Memlet(f"_b_bn[0:{K}, {i}]"))
                else:
                    # Internally, the MV will take care of properly reading B as the vector x
                    state.add_memlet_path(B, gemv_node, dst_conn="_x", memlet=Memlet(f"_b_bn[0:{BS}, 0:{K}, {i}]"))
            state.add_memlet_path(gemv_node, tmp, src_conn="_y", memlet=Memlet(f"_tmp[0:{N}, {i}]"))

        # Gather (fake gather) and output with the right pattern, in this case is row by row
        libnode = Gather('Gather_MMM', row_major=True)

        libnode.implementation = "gather"
        state.add_node(libnode)

        state.add_edge(tmp, None, libnode, '_in', dace.Memlet.from_array('_tmp', array_tmp))
        state.add_edge(libnode, '_out', C, None, dace.Memlet.from_array('_c', array_c))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandMMM_MV.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandMMM_OP_Col(ExpandTransformation):
    '''
    In this expansion we have as many OP as K (the number of columns of matrix A).
    Each one of them will take in input a column of A, a row of B and computes the outer product (assuming none of them is transposed).
    The results are combined together (summed) and the final matrix C is output in a column major order

    Both A and B are read multiple times so they are stored in buffer nodes.
    
    TODO: factorize code to support Col and Row major ordering
    '''

    environments = []

    supported_flags: Set[str] = {'op'}

    @staticmethod
    def can_be_used(flags: Set[str]):
        """
        Returns whether this expansion supports at least one of the flags provided as arguments.
        """
        return bool(flags & ExpandMMM_OP_Col.supported_flags)

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b, shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_b).type]

        is_batched = node.batched if hasattr(node, 'batched') else False  # Used only with ONNX Matmul

        ### Transposition currently not supported
        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        # Deal with degenerated cases (N or M equal to 1)
        if len(trans_shape_a) == 1:
            N = 1
            K = trans_shape_a[0]

            shape_a = (N, K)
            if len(strides_a) == 0:
                strides_a = [K, 1]
            else:
                strides_a = [K, strides_a[-1]]
        else:
            N = trans_shape_a[-2]
            K = trans_shape_a[-1]

        if len(trans_shape_b) == 1:
            M = 1
            if node.transB:
                shape_b = (M, K)
                if len(strides_b) == 0:
                    strides_b = [K, 1]
                else:
                    strides_b = [K, strides_a[-1]]

            else:
                shape_b = (K, M)
                if len(strides_b) == 0:
                    strides_b = [M, 1]
                else:
                    strides_b = [M, strides_a[-1]]

        else:
            M = trans_shape_b[-1]

        if is_batched:
            # In this case there is another dimension
            BS = trans_shape_b[0]
        else:
            BS = 1

        if N == 1 or M == 1:
            strides_c = [M, *cdata[-1]]
        else:
            strides_c = cdata[-1]

        if node.transA:
            raise NotImplementedError("Expansion does not support transposed matrix")

        shape_c = (N, M)

        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=strides_c, storage=cdata[1].storage)

        if node.alpha == 1.0:
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))

        state = sdfg.add_state(node.label + "_state")

        # In this case A and B must be always in a buffer node. We add it explicitly to facilitate canonical DAG construction
        # (can be removed later on with a RedundantArray transformation)
        # TODO: have a way to identify this nodes properly, like properties of the expansion

        _, array_a_bn = sdfg.add_array("_a_bn",
                                       shape_a,
                                       dtype_a,
                                       strides=strides_a,
                                       storage=outer_array_a.storage,
                                       transient=True)

        _, array_b_bn = sdfg.add_array("_b_bn",
                                       shape_b,
                                       dtype_b,
                                       strides=strides_b,
                                       storage=outer_array_b.storage,
                                       transient=True)
        A_in = state.add_read("_a")
        A = state.add_access("_a_bn")
        state.add_edge(A_in, None, A, None, dace.Memlet.from_array('_a', array_a))

        B_in = state.add_read("_b")
        B = state.add_access("_b_bn")
        state.add_edge(B_in, None, B, None, dace.Memlet.from_array('_b', array_b))

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        ## Computation State

        # Each OP will produce a transient result, and all of them will be combined (summed) together
        # Note: an option would be to use a reduction tree
        # In this case we will create a transient with shape [K, N, M], and later one we MMMReduceMMM

        _, partial_resuls_arr = sdfg.add_transient("_partial_results", (K, N, M), dtype_c)

        partial_results = state.add_access("_partial_results")
        wC = state.add_write("_c")

        # Add the OP library nodes
        op_implementation = "OP_by_col"

        for k in range(K):
            op_node = OP("op", alpha=node.alpha, used_in_batched_matmul=is_batched)
            op_node.implementation = op_implementation
            state.add_memlet_path(A, op_node, dst_conn="_u", memlet=Memlet(f"_a_bn[0:{N}, {k}]"))
            if not node.transB:
                # use a row of B
                if not is_batched:
                    state.add_memlet_path(B, op_node, dst_conn="_v", memlet=Memlet(f"_b_bn[{k}, 0:{M}]"))
                else:
                    state.add_memlet_path(B, op_node, dst_conn="_v", memlet=Memlet(f"_b_bn[0:{BS}, {k}, 0:{M}]"))
            else:
                # use a column of B
                assert not is_batched
                state.add_memlet_path(B, op_node, dst_conn="_v", memlet=Memlet(f"_b_bn[0:{M}, {k}]"))

            state.add_memlet_path(op_node,
                                  partial_results,
                                  src_conn="_A",
                                  memlet=Memlet(f"_partial_results[{k}, 0:{N}, 0:{M}]"))

        reduce_libnode = ReduceMMM('Reduce')

        reduce_libnode.implementation = "reduce_sum_col"
        state.add_node(reduce_libnode)
        # BAD Trick to have the right volume
        # state.add_edge(partial_results,
        #                None,
        #                reduce_libnode,
        #                '_in',
        #                memlet=Memlet(f"_partial_results[0, 0:{N}, 0:{M}]"))
        state.add_edge(partial_results, None, reduce_libnode, '_in',
                       dace.Memlet.from_array("_partial_results", partial_resuls_arr))
        state.add_edge(reduce_libnode, '_out', wC, None, dace.Memlet.from_array("_c", array_c))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandMMM_OP_Col.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandMMM_LMV(ExpandTransformation):
    '''
    In this expansion we have as many Left-Matrix Vector multiplication N (the number of rows of matrix C).
    Each one of them will take in input a row of A, the entire B (read by column) and produces a row of C.

    B can be broadcasted on the fly, while A needs to be buffered.

    In this implementation, B is read by columns so that C is produced by columns
    '''

    environments = []

    supported_flags: Set[str] = {'lmv'}

    @staticmethod
    def can_be_used(flags: Set[str]):
        """
        Returns whether this expansion supports at least one of the flags provided as arguments.
        """
        return bool(flags & ExpandMMM_LMV.supported_flags)

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        sdfg = dace.SDFG(node.label + "_sdfg")

        ((edge_a, outer_array_a, shape_a, strides_a), (edge_b, outer_array_b, shape_b, strides_b),
         cdata) = _get_matmul_operands(node, parent_state, parent_sdfg)

        dtype_a = outer_array_a.dtype.type
        dtype_b = outer_array_b.dtype.type
        dtype_c = dace.DTYPE_TO_TYPECLASS[np.result_type(dtype_a, dtype_b).type]

        is_batched = node.batched if hasattr(node, 'batched') else False  # Used only with ONNX Matmul

        ### Transposition currently not supported
        if node.transA:
            trans_shape_a = list(reversed(shape_a))
        else:
            trans_shape_a = shape_a

        if node.transB:
            trans_shape_b = list(reversed(shape_b))
        else:
            trans_shape_b = shape_b

        # if (len(trans_shape_a) != 2 or len(trans_shape_b) != 2 or trans_shape_a[1] != trans_shape_b[0]):
        #     raise SyntaxError("Matrix sizes must match")

        # one dimension could be one
        if len(trans_shape_a) == 1:
            N = 1
            K = trans_shape_a[0]
        else:
            N = trans_shape_a[-2]
            K = trans_shape_a[-1]
        if len(trans_shape_b) == 1:
            M = 1
        else:
            M = trans_shape_b[-1]

        if is_batched:
            # In this case there is another dimension
            BS = trans_shape_b[0]
            num_lmv = N // BS  # consider the original N, not the flatted dimension
        else:
            BS = 1
            num_lmv = N

        shape_c = (N, M)

        storage = outer_array_a.storage

        _, array_a = sdfg.add_array("_a", shape_a, dtype_a, strides=strides_a, storage=outer_array_a.storage)
        _, array_b = sdfg.add_array("_b", shape_b, dtype_b, strides=strides_b, storage=outer_array_b.storage)
        _, array_c = sdfg.add_array("_c", shape_c, dtype_c, strides=cdata[-1], storage=cdata[1].storage)

        if node.alpha == 1.0:
            mul_program = "__out = __a * __b"
        else:
            mul_program = "__out = {} * __a * __b".format(_cast_to_dtype_str(node.alpha, dtype_a))

        # init_state = sdfg.add_state(node.label + "_initstate")
        # state = sdfg.add_state_after(init_state, node.label + "_state")
        state = sdfg.add_state(node.label + "_state")

        # In this case A must be always in a buffer node. We add it explicitly to facilitate canonical DAG construction
        # (can be removed later on with a RedundantArray transformation)
        # TODO: have a way to identify this nodes properly, like properties of the expansion

        _, array_a_bn = sdfg.add_array("_a_bn",
                                       shape_a,
                                       dtype_a,
                                       strides=strides_a,
                                       storage=outer_array_a.storage,
                                       transient=True)

        A_in = state.add_read("_a")
        A = state.add_access("_a_bn")
        state.add_edge(A_in, None, A, None, dace.Memlet.from_array('_a', array_a))

        mul_out, mul_out_array = "_c", array_c
        output_nodes = None

        ## Computation State
        B = state.add_read("_b")

        # Broadcast (fake broadcast) node that will broadcast B to all the MLV nodes (see below).
        # This is currently implemented as a copy to a transient, and we do this to favor streaming composition (we are interested)
        # in the access patter to B, that in this case is by column
        _, array_broadc_B = sdfg.add_array("_broadcasted_B",
                                           shape_b,
                                           dtype_b,
                                           transient=True,
                                           strides=strides_b,
                                           storage=outer_array_b.storage)
        broadcast_B = state.add_access("_broadcasted_B")

        if not node.transB:
            broadcast_libnode = Broadcast('Broadcast_B_MMM', row_major=False, used_in_batched_matmul=is_batched)
        else:
            # this favors streamability
            broadcast_libnode = Broadcast('Broadcast_B_MMM', row_major=True)

        broadcast_libnode.implementation = "broadcast"
        state.add_node(broadcast_libnode)

        state.add_edge(B, None, broadcast_libnode, '_in', dace.Memlet.from_array('_b', array_b))
        state.add_edge(broadcast_libnode, '_out', broadcast_B, None,
                       dace.Memlet.from_array('_broadcasted_B', array_broadc_B))

        # To favor streaming composition, we need to add a transient, that is used to combine things, and then we copy
        # to the final container using the exact pattern that would be used in the actual computation
        # In this case, the final matrix C will be produced column by column
        _, array_tmp = sdfg.add_array("_tmp",
                                      shape_c,
                                      dtype_c,
                                      transient=True,
                                      strides=cdata[-1],
                                      storage=cdata[1].storage)
        tmp = state.add_access("_tmp")
        C = state.add_write("_c")

        # Add the LMV library nodes

        lmv_implementation = "LMV_col"
        # Note: memlet to MV already account for the right data volume

        # NOTE, for batched matmul. With the LMV approach, each LMV node will write a row of A.
        # If we are working with batched matmul, the LMV node `i` will be in charge of computing multiple rows. If C is BxNxM, it will
        # compute row [0,i,:], [1,i,:]  ..., [B-1,i,:] (note that this is actually flattened)
        # Differently from the MV or OP approach, here is the LMV node that must know its `i` (for example, with MV, every node writes an entire colum and this is
        # know while building them, and we set the appropriate memlet).
        # Therefore this must be handled withing the library node

        for i in range(num_lmv):
            # Note: if B is transposed then it changes the way in which we compute the LMV
            # (in the MV version is not the case as B will contributed only to the vector of the MV)
            lmv_node = LMV("lmv", alpha=node.alpha, beta=0, transA=node.transB, used_in_batched_matmul=is_batched, id=i)
            lmv_node.implementation = lmv_implementation
            # A single row of A is multiplied with  whole B (the matrix in the LMV)
            if not is_batched:
                state.add_memlet_path(A, lmv_node, dst_conn="_x", memlet=Memlet(f"_a_bn[{i}, 0:{K}]"))
            else:
                state.add_memlet_path(A, lmv_node, dst_conn="_x", memlet=Memlet(f"_a_bn[0:{N}, 0:{K}]"))
            if not node.transB:
                if not is_batched:
                    state.add_memlet_path(broadcast_B,
                                          lmv_node,
                                          dst_conn="_A",
                                          memlet=Memlet(f"_broadcasted_B[0:{K}, 0:{M}]"))
                else:
                    state.add_memlet_path(broadcast_B,
                                          lmv_node,
                                          dst_conn="_A",
                                          memlet=Memlet(f"_broadcasted_B[0:{BS}, 0:{K}, 0:{M}]"))
            else:
                # TODO: is this the right place to do this, or in the Broadcast node?

                state.add_memlet_path(broadcast_B,
                                      lmv_node,
                                      dst_conn="_A",
                                      memlet=Memlet(f"_broadcasted_B[0:{M}, 0:{K}]"))
            if not is_batched:
                state.add_memlet_path(lmv_node, tmp, src_conn="_y", memlet=Memlet(f"_tmp[{i}, 0:{M}]"))
            else:
                state.add_memlet_path(lmv_node, tmp, src_conn="_y", memlet=Memlet(f"_tmp[0:{N}, 0:{M}]"))

        # Copy out with the right pattern. In this case is column by column
        # Gather (fake gather) and output with the right pattern, in this case is row by row
        libnode = Gather('Gather_MMM', row_major=False)

        libnode.implementation = "gather"
        state.add_node(libnode)

        state.add_edge(tmp, None, libnode, '_in', dace.Memlet.from_array('_tmp', array_tmp))
        state.add_edge(libnode, '_out', C, None, dace.Memlet.from_array('_c', array_c))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandMMM_LMV.make_sdfg(node, state, sdfg)


@dace.library.node
class MMM(dace.sdfg.nodes.LibraryNode):
    """Executes C = A @ B
    """

    # Global properties
    implementations = {"mv": ExpandMMM_MV, "op_col": ExpandMMM_OP_Col, "LMV_col": ExpandMMM_LMV}
    default_implementation = "mv"

    # Object fields
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    batched = properties.Property(dtype=bool, desc="Batched MatMul (Used for ONNX MatMul node only)")

    def __init__(self, name, location=None, transA=False, transB=False, alpha=1, batched=False):
        super().__init__(name, location=location, inputs=({"_a", "_b"}), outputs={"_c"})
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.batched = batched

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [2]:
            raise ValueError("Expected 2 inputs to MMM")
        size2 = None
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a':
                subset_a = dc(memlet.subset)
                subset_a.squeeze()
                size0 = subset_a.size()
            if dst_conn == '_b':
                subset_b = dc(memlet.subset)
                subset_b.squeeze()
                size1 = subset_b.size()
            if dst_conn == '_c':
                subset_c = dc(memlet.subset)
                subset_c.squeeze()
                size2 = subset_c.size()

        if self.transA:
            size0 = list(reversed(size0))
        if self.transB:
            size1 = list(reversed(size1))

        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        # Function is symmetric, edge order does not matter
        if not self.batched and (len(size0) > 2 or len(size1) > 2):  # one among N, and  M can also be one
            raise ValueError("matrix-matrix product only supported on matrices")
        if self.batched and (len(size0) > 3 or len(size1) > 3):  # one among N, and  M can also be one
            raise ValueError("Batched matrix-matrix product only supported on 3D matrices")
        if not self.batched:
            if (len(size0) == 2 and size0[1] != size1[0]) or (len(size0) == 1 and size0[0] != size1[0]):
                raise ValueError("Inputs to matrix-matrix product " "must agree in the k-dimension")
        else:
            if size0[-1] != size1[-2]:
                raise ValueError("Inputs to matrix-matrix product " "must agree in the k-dimension")

        out_subset = dc(out_memlet.subset)
        out_subset.squeeze()
        size3 = out_subset.size()
        if size2 is not None and size2 != size3:
            raise ValueError("Input C matrix must match output matrix.")
        if not self.batched and len(size3) > 2:
            raise ValueError("matrix-matrix product only supported on matrices")
        if len(size3) == 2 and list(size3) != [size0[-2], size1[-1]]:
            raise ValueError("Output to matrix-matrix product must agree in the m and n " "dimensions")


#####################################################################################
# END Expansions
#####################################################################################
