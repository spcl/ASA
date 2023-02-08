# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
   Tests for matrix-matrix multiplication library node
"""

import dace
import numpy as np
import pytest
from canonical_libnodes.blas.mmm import MMM
from dace.transformation.transformation import ExpandTransformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG


def create_mmm_sdfg(dtype, A_shape, B_shape, C_shape, transA, transB, alpha, beta, implementation, sdfg_name):
    ''' 
        Computes C = A@B
    '''

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    B, B_arr = sdfg.add_array("B", B_shape, dtype)
    C, C_arr = sdfg.add_array("C", C_shape, dtype)

    rA = state.add_read("A")
    rB = state.add_read("B")
    wC = state.add_access("C")
    libnode = MMM('MMM', transA=transA, transB=transB, alpha=alpha)

    ##### ADD custom expansion to existing Library Node
    #Gemm.register_implementation("my_own", ExpandMMMPure)

    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_b', dace.Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_c', wC, None, dace.Memlet.from_array(C, C_arr))
    if beta != 0.0:
        raise RuntimeError("Not supported")
        rC = state.add_read('C')
        state.add_edge(rC, None, libnode, '_cin', dace.Memlet.from_array(C, C_arr))

    return sdfg


def run_test(implementation="pure", N=4, K=4, M=4, transB=False, np_dtype=np.float32):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{N}_{M}_{K}"

    # shape of the transposed arrays
    A_shape = [N, K]
    if not transB:
        B_shape = [K, M]
    else:
        B_shape = [M, K]
    C_shape = [N, M]

    ## Transp not currently supported
    print(f'Matrix multiplication {N}x{K}x{M}')

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    B = np.random.rand(*B_shape).astype(np_dtype)
    if C_shape is not None:
        C = np.random.rand(*C_shape).astype(np_dtype)
    else:
        C = None
    C = np.zeros(C_shape, dtype=np_dtype)

    def numpy_gemm(A, B, C, transB):
        if transB:
            return A @ B.T
        else:
            return A @ B

    regression = numpy_gemm(A, B, C, transB)

    sdfg = create_mmm_sdfg(np_dtype, A_shape, B_shape, C_shape, False, transB, 1, 0, implementation, sdfg_name)

    # from dse.dse import DSE
    # DSE(sdfg)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, regression)


def specialize_matmul(sdfg: dace.SDFG):
    '''
    Given a one-level SDFG, specializes all the matmul nodes (by expanding them)
    '''
    assert len(sdfg.nodes()) == 1
    state = sdfg.nodes()[0]
    for node in state.nodes():
        if isinstance(node, dace.libraries.blas.MatMul):
            node.expand(sdfg, state)


def run_dace_program(implementation="canonical_mv", N=4, K=4, M=4, transB=False):
    # use a dace program and then expand the matmul with one of our
    # canonical expansions

    @dace.program
    def mmm_test(A: dace.float32[N, K], B: dace.float32[K, M], C: dace.float32[N, M]):
        C[:] = A @ B

    # shape of the transposed arrays
    A_shape = [N, K]
    if not transB:
        B_shape = [K, M]
    else:
        B_shape = [M, K]
    C_shape = [N, M]

    ## Transp not currently supported
    print(f'Matrix multiplication {N}x{K}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    B = np.random.rand(*B_shape).astype(np_dtype)
    if C_shape is not None:
        C = np.random.rand(*C_shape).astype(np_dtype)
    else:
        C = None
    C = np.zeros(C_shape, dtype=np_dtype)

    def numpy_gemm(A, B, C, transB):
        if transB:
            return A @ B.T
        else:
            return A @ B

    regression = numpy_gemm(A, B, C, transB)

    sdfg = mmm_test.to_sdfg()

    # TODO: register expansion
    specialize_matmul(sdfg)
    gemm_node = sdfg.nodes()[0].nodes()[-1]
    gemm_node.implementation = implementation

    sdfg(A=A, B=B, C=C)
    assert np.allclose(C, regression)


@pytest.mark.parametrize("implementation, N, K, M, transB", [("mv", 4, 4, 5, False), ("mv", 1, 4, 5, False),
                                                             ("mv", 4, 4, 5, True), ("mv", 5, 7, 1, True),
                                                             ("op_col", 4, 4, 5, False), ("op_col", 4, 4, 5, True),
                                                             ("LMV_col", 4, 4, 5, False), ("LMV_col", 5, 7, 4, True)])
def test_mmm(implementation, N, K, M, transB):
    run_test(implementation=implementation, N=N, K=K, M=M, transB=transB)


if __name__ == "__main__":
    # run_test(implementation="mv", np_dtype=np.complex64)
    # run_test(implementation="mv", N=4, K=4, M=5)
    # run_test(implementation="mv", N=5, K=7, M=4)
    # run_test(implementation="mv", N=1, K=4, M=5)  # degenerated

    # # # with transposed B
    # run_test(implementation="mv", N=4, K=4, M=5, transB=True)
    # run_test(implementation="mv", N=5, K=7, M=4, transB=True)
    # run_test(implementation="mv", N=5, K=7, M=1, transB=True)  # degenerated

    # Outer product, result produce in column major
    run_test(implementation="op_col")
    run_test(implementation="op_col", N=4, K=4, M=5)
    run_test(implementation="op_col", N=5, K=7, M=4)
    # run_test(implementation="op_col", N=1, K=7, M=4)  # degenerated, does not work
    run_test(implementation="op_col", N=4, K=4, M=5, transB=True)
    run_test(implementation="op_col", N=5, K=7, M=4, transB=True)
    # # TODO: add another node after, add combiners?

    run_test(implementation="LMV_col")
    run_test(implementation="LMV_col", N=4, K=4, M=5)
    run_test(implementation="LMV_col", N=5, K=7, M=4)
    run_test(implementation="LMV_col", N=4, K=4, M=5, transB=True)
    run_test(implementation="LMV_col", N=5, K=7, M=4, transB=True)

    # run_dace_program()
    # run_dace_program(implementation="canonical_lmv", N=5, K=7, M=4)
    # run_dace_program(implementation="canonical_op_col", N=5, K=7, M=4)
    # run_dace_program(implementation="canonical_mv", N=8, K=8, M=1)
