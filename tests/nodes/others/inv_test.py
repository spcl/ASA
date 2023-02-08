# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
  Test for Matrix inversion canonical expansion
"""
import dace
import numpy as np
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes.others.inv import ExpandInvDummy, ExpandInvCholeskyFwdTRMM

N = dace.symbol('N', dace.int32)
N = 4


@dace.program
def inv(A: dace.float64[N, N]):
    A_inv = np.linalg.inv(A)
    return A_inv


def manual_inv(A, N):
    cholesky = np.zeros([N, N]).astype(np.float64)
    for row in range(N):
        for col in range(row + 1):
            s = 0
            for k in range(col):
                s += cholesky[row, k] * cholesky[col, k]
            if row == col:
                cholesky[row, col] = np.sqrt(A[row, col] - s)
            else:
                cholesky[row, col] = (A[row, col] - s) / cholesky[col, col]

    # print(cholesky)
    # AA = np.linalg.cholesky(A)
    # print(AA)
    # import pdb
    # pdb.set_trace()

    fs = np.zeros([N, N])
    print(cholesky)

    # for col in range(N):
    #     for row in range(col, N):
    # TRSM, solves LU=I, where U is triangular lower
    for row in range(N):
        for col in range(row + 1):
            s = 0
            if row == col:
                fs[row, col] = 1 / cholesky[row, col]
            else:
                for k in range(col, row):
                    s += cholesky[row, k] * fs[k, col]

                fs[row, col] = -s / cholesky[row, row]
    print(fs)

    inverted = fs.T @ fs
    return inverted


def test_inversion():

    # generate symmetric definite positive matrix (Note: if used with lower precision may produce small error)
    inp = np.random.rand(N, N).astype(np.float64)
    A = inp.T @ inp

    # Actual computation using the right algorithm
    A_ref = np.linalg.inv(A)

    sdfg = inv.to_sdfg(simplify=True)
    dace.libraries.linalg.Inv.register_implementation("canonical", ExpandInvCholeskyFwdTRMM)
    sdfg.nodes()[0].nodes()[-1].implementation = "canonical"

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)

    A_dace = sdfg(A=A)
    diff = np.linalg.norm(A_dace - A_ref) / np.linalg.norm(A_ref)
    assert np.allclose(A_dace, A_ref)
