# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Test for Forward Substitution canonical expansion
"""

import dace
import numpy as np
from canonical_libnodes.others.forward_substitution import ForwardSubstitution
from dace.transformation.interstate import InlineSDFG


def ref_ffs(A, N):
    fs = np.zeros([N, N])

    # for col in range(N):
    #     for row in range(col, N):
    # TRSM, solves LU=I, where U is triangular lower
    for row in range(N):
        for col in range(row + 1):
            s = 0
            if row == col:
                fs[row, col] = 1 / A[row, row]
            else:
                for k in range(col, row):
                    s += A[row, k] * fs[k, col]

                fs[row, col] = -s / A[row, row]
    return fs


def create_fs_sdfg(dtype, A_shape, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    B, B_arr = sdfg.add_array("B", A_shape, dtype)

    rA = state.add_read("A")
    wB = state.add_write("B")
    libnode = ForwardSubstitution('fs')

    libnode.implementation = "seq"
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a', dace.Memlet.from_array(A, A_arr))
    state.add_edge(libnode, '_b', wB, None, dace.Memlet.from_array(B, B_arr))

    return sdfg


def test_forward_substitution():
    N = 8
    # generate symmetric definite positive matrix
    rand = np.random.rand(N, N).astype(np.float64)
    A = rand.T @ rand

    ref = ref_ffs(A, N)

    sdfg = create_fs_sdfg(np.float64, [N, N], "forward_substitution_ln")
    B = np.random.rand(N, N).astype(np.float64)
    sdfg(A=A, B=B)
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG()], print_report=True)

    assert np.allclose(B, ref)