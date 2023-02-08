# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
   Tests for matrix-vector multiplication library node
"""

from canonical_libnodes.blas.mv import MV
import dace
import numpy as np
from dace.transformation.transformation import ExpandTransformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG
import pytest


def create_mv_sdfg(dtype, A_shape, x_shape, y_shape, transA, alpha, beta, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    x, x_arr = sdfg.add_array("x", x_shape, dtype)
    y, y_arr = sdfg.add_array("y", y_shape, dtype)

    rA = state.add_read("A")
    rx = state.add_read("x")
    wy = state.add_write("y")
    libnode = MV('MV', transA=transA, alpha=alpha, beta=0)

    ##### ADD custom expansion to existing Library Node
    #Gemm.register_implementation("my_own", ExpandMMMPure)

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


def run_test(implementation="pure", N=4, M=4):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{N}_{M}"

    # shape of the transposed arrays
    A_shape = [N, M]
    x_shape = [M]
    y_shape = [N]

    ## Transp not currently supported
    print(f'Matrix-Vector multiplication {N}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    x = np.random.rand(*x_shape).astype(np_dtype)
    y = np.random.rand(*y_shape).astype(np_dtype)

    y_regression = A @ x

    sdfg = create_mv_sdfg(dace.float32, A_shape, x_shape, y_shape, False, 1, 0, implementation, sdfg_name)

    # from converter.converter import convert_sdfg_to_dag

    # convert_sdfg_to_dag(sdfg)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineMultistateSDFG], print_report=True)
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    sdfg(A=A, x=x, y=y)

    # diff = np.linalg.norm(C_regression - C) / (M * N)
    # print("Difference:", diff)
    # assert diff <= 1e-5
    assert np.allclose(y, y_regression)


@pytest.mark.parametrize("implementation, N, M", [("seq", 3, 5), ("seq", 16, 7), ("seq", 1, 5)])
def test_mv(implementation, N, M):
    run_test(implementation=implementation, N=N, M=M)


if __name__ == "__main__":
    run_test(implementation="seq", N=3, M=5)
    run_test(implementation="seq", N=16, M=7)
    run_test(implementation="seq", N=1, M=5)