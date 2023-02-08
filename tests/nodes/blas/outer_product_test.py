# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
   Tests for outer product library node
"""

import pytest
import dace
import numpy as np
from canonical_libnodes.blas.op import OP
from dace.transformation.transformation import ExpandTransformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG


def create_op_sdfg(dtype, u_shape, v_shape, A_shape, alpha, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    u, u_arr = sdfg.add_array("u", u_shape, dtype)
    v, v_arr = sdfg.add_array("v", v_shape, dtype)
    A, A_arr = sdfg.add_array("A", A_shape, dtype)

    ru = state.add_read("u")
    rv = state.add_read("v")
    wA = state.add_write("A")
    libnode = OP('OP', alpha=alpha)

    ##### ADD custom expansion to existing Library Node
    #Gemm.register_implementation("my_own", ExpandMMMPure)

    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(ru, None, libnode, '_u', dace.Memlet.from_array(u, u_arr))
    state.add_edge(rv, None, libnode, '_v', dace.Memlet.from_array(v, v_arr))
    state.add_edge(libnode, '_A', wA, None, dace.Memlet.from_array(A, A_arr))

    return sdfg


def run_test(implementation="OP_by_col", N=4, M=5):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{N}_{M}"

    # shape of the transposed arrays
    A_shape = [N, M]
    u_shape = [N]
    v_shape = [M]

    ## Transp not currently supported
    print(f'Outer product {N}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    A = np.random.rand(*A_shape).astype(np_dtype)
    u = np.random.rand(*u_shape).astype(np_dtype)
    v = np.random.rand(*v_shape).astype(np_dtype)

    A_regression = np.outer(u, v)

    sdfg = create_op_sdfg(dace.float32, u_shape, v_shape, A_shape, 1, implementation, sdfg_name)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    # sdfg.view()
    sdfg(A=A, u=u, v=v)

    assert np.allclose(A, A_regression)


@pytest.mark.parametrize("implementation, N, M", [("OP_by_col", 4, 5), ("OP_by_col", 7, 4), ("OP_by_col", 1, 5),
                                                  ("OP_by_col", 4, 1), ("OP_by_row", 7, 4), ("OP_by_row", 1, 4),
                                                  ("OP_by_row", 4, 1)])
def test_op(implementation, N, M):
    run_test(implementation=implementation, N=N, M=M)


if __name__ == "__main__":
    run_test("OP_by_col", 4, 5)
    run_test("OP_by_col", 7, 4)
    run_test("OP_by_col", 1, 5)
    run_test("OP_by_col", 4, 1)
    run_test("OP_by_row", 7, 4)
    run_test("OP_by_row", 1, 4)
    run_test("OP_by_row", 4, 1)
