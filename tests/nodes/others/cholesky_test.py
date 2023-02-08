# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Cholesky canonical expansion test
"""
import dace
import numpy as np
from canonical_libnodes.others.cholesky import ExpandCholeskySeq
from dace.transformation.interstate import InlineSDFG

N = dace.symbol("N", dace.int32)


@dace.program
def cholesky(A: dace.float64[N, N]):
    return np.linalg.cholesky(A)


def manual_cholesky(sdfg_name, A_shape, dtype):
    """
        Not actually used, left here for reference
    """
    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", A_shape, dtype)
    R, R_Arr = sdfg.add_array("R", A_shape, dtype)  # result

    N, M = A_shape
    assert N == M  # should be a squared matrix

    # Zeroing the result

    zeroing_map_entry, zeroing_map_exit = state.add_map("zero_map", {
        "i": f"0:{N}",
        "j": f"0:{N}"
    },
                                                        schedule=dace.ScheduleType.Default)
    init_R = state.add_access("R")
    zeroing_tasklet = state.add_tasklet("zeroing", {}, {"R_out"}, "R_out = 0")
    state.add_memlet_path(zeroing_map_entry, zeroing_tasklet, memlet=dace.Memlet())
    state.add_memlet_path(zeroing_tasklet, zeroing_map_exit, init_R, src_conn="R_out", memlet=dace.Memlet(f"R[i,j]"))

    ## Computational maps
    comp_map_entry, comp_map_exit = state.add_map("comp_map", {
        "row": f"0:{N}",
        "col": f"0:row+1"
    },
                                                  schedule=dace.ScheduleType.Sequential)

    ## init s
    sdfg.add_scalar("s", dtype, storage=dace.StorageType.Register, transient=True)
    s_access_in = state.add_access("s")
    s_access_out = state.add_access("s")

    init_s = state.add_tasklet("init_s", {}, {"s_out"}, "s_out = 0")

    state.add_memlet_path(comp_map_entry, init_s, memlet=dace.Memlet())
    state.add_memlet_path(init_s, s_access_in, src_conn="s_out", memlet=dace.Memlet(f"s[0]"))

    ## K map, accumulate over s
    # for k in range(col):
    #     s += cholesky[row, k] * cholesky[col, k]

    k_map_entry, k_map_exit = state.add_map("k_map", {
        "k": f"0:col",
    }, schedule=dace.ScheduleType.Sequential)
    multiply_tasklet = state.add_tasklet("multiply", {"chol_row_k", "chol_col_k", "s_in"}, {"s_out"},
                                         "s_out = chol_row_k * chol_col_k + s_in")
    # inputs
    state.add_memlet_path(init_R,
                          comp_map_entry,
                          k_map_entry,
                          multiply_tasklet,
                          dst_conn="chol_row_k",
                          memlet=dace.Memlet("R[row, k]"))
    state.add_memlet_path(init_R,
                          comp_map_entry,
                          k_map_entry,
                          multiply_tasklet,
                          dst_conn="chol_col_k",
                          memlet=dace.Memlet("R[col, k]"))
    state.add_memlet_path(s_access_in, k_map_entry, multiply_tasklet, dst_conn="s_in", memlet=dace.Memlet(f"s[0]"))
    #output
    state.add_memlet_path(multiply_tasklet, k_map_exit, s_access_out, src_conn="s_out", memlet=dace.Memlet(f"s[0]"))

    ## Write tasklet
    # if row == col:
    #     cholesky[row, col] = np.sqrt(A[row, col] - s)
    # else:
    #     cholesky[row, col] = (A[row, col] - s) / cholesky[col, col]

    # TODO: we don't need to read the diagonal element everytime
    write_out_tasklet = state.add_tasklet(
        "write_out", {"A_row_col", "s_in", "R_col_col"}, {"R_out"}, """
if row == col:
    R_out = sqrt(A_row_col - s_in)
else:
    R_out = (A_row_col - s_in) / R_col_col
""")

    # inputs
    A_read = state.add_read("A")
    state.add_memlet_path(s_access_out, write_out_tasklet, dst_conn="s_in", memlet=dace.Memlet(f"s[0]"))
    state.add_memlet_path(A_read,
                          comp_map_entry,
                          write_out_tasklet,
                          dst_conn="A_row_col",
                          memlet=dace.Memlet(f"A[row, col]"))

    state.add_memlet_path(init_R,
                          comp_map_entry,
                          write_out_tasklet,
                          dst_conn="R_col_col",
                          memlet=dace.Memlet("R[col, col]"))

    # output
    R_write = state.add_write("R")
    state.add_memlet_path(write_out_tasklet,
                          comp_map_exit,
                          R_write,
                          src_conn="R_out",
                          memlet=dace.Memlet("R[row, col]"))

    return sdfg


def test_cholesky():
    N = 8
    # generate symmetric definite positive matrix
    rand = np.random.rand(N, N).astype(np.float64)
    A = rand.T @ rand

    ref = np.linalg.cholesky(A)

    from dace.libraries import linalg

    linalg.nodes.cholesky.Cholesky.register_implementation("seq", ExpandCholeskySeq)
    sdfg = cholesky.to_sdfg()
    chol = sdfg.nodes()[0].nodes()[-1]
    chol.implementation = "seq"

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG()], print_report=True)

    dace_result = sdfg(A, N=N)

    # Check that the result is correct
    assert np.allclose(dace_result, ref)
