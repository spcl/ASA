# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
   Tests for left-matrix-vector multiplication library node
"""

from canonical_libnodes.blas.lmv import LMV
import dace
import numpy as np
import pytest
from dace.transformation.transformation import ExpandTransformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG


def create_lmv_sdfg(dtype, A_shape, x_shape, y_shape, transA, alpha, beta, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    x, x_arr = sdfg.add_array("x", x_shape, dtype)
    y, y_arr = sdfg.add_array("y", y_shape, dtype)

    rA = state.add_read("A")
    rx = state.add_read("x")
    wy = state.add_write("y")
    libnode = LMV('LMV', transA=transA, alpha=alpha, beta=0)

    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_A', dace.Memlet.from_array(A, A_arr))
    state.add_edge(rx, None, libnode, '_x', dace.Memlet.from_array(x, x_arr))
    state.add_edge(libnode, '_y', wy, None, dace.Memlet.from_array(y, y_arr))
    if beta != 0.0:
        raise RuntimeError("Not supported")
        rC = state.add_read('C')
        state.add_edge(rC, None, libnode, '_cin', dace.Memlet.from_array(C, C_arr))
    return sdfg


def run_test(implementation="LMV_col", K=4, M=4, transA=False):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{K}_{M}"

    # shape of the transposed arrays
    if not transA:
        A_shape = [K, M]
    else:
        A_shape = [M, K]
    x_shape = [K]
    y_shape = [M]

    ## Transp not currently supported
    print(f'Left Matrix-Vector multiplication {K}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    x = np.random.rand(*x_shape).astype(np_dtype)
    y = np.random.rand(*y_shape).astype(np_dtype)

    if not transA:
        y_regression = x @ A
    else:
        y_regression = x @ A.T

    sdfg = create_lmv_sdfg(dace.float32, A_shape, x_shape, y_shape, transA, 1, 0, implementation, sdfg_name)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineMultistateSDFG], print_report=True)
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    sdfg(A=A, x=x, y=y)

    assert np.allclose(y, y_regression)


@pytest.mark.parametrize("K, M, transA", [(4, 4, False), (2, 5, False), (7, 4, False), (4, 4, True), (2, 5, True)])
def test_lmv(K, M, transA):
    run_test(K=K, M=M, transA=transA)


if __name__ == "__main__":
    run_test()
    run_test(K=2, M=5)
    run_test(K=7, M=4)
    run_test(transA=True)
    run_test(K=2, M=5, transA=True)