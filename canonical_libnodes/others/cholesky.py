# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Pure and sequential implementatin for Cholesky factorization
'''
import dace
from dace.transformation.transformation import ExpandTransformation
from dace import library


@dace.library.expansion
class ExpandCholeskySeq(ExpandTransformation):
    '''
    Sequential Cholesky implementation
    for row in range(N):
        for col in range(row + 1):
            s = 0
            for k in range(col):
                s += cholesky[row, k] * cholesky[col, k]
            if row == col:
                cholesky[row, col] = np.sqrt(A[row, col] - s)
            else:
                cholesky[row, col] = (A[row, col] - s) / cholesky[col, col]
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        inp_desc, inp_shape, out_desc, out_shape = node.validate(parent_sdfg, parent_state)
        dtype = inp_desc.dtype

        ain_arr = sdfg.add_array('_a', inp_shape, dtype=dtype, strides=inp_desc.strides)
        bout_arr = sdfg.add_array('_b', out_shape, dtype=dtype, strides=out_desc.strides)

        N, M = inp_shape
        assert N == M  # should be a squared matrix
        assert node.lower

        # Zeroing the result

        zeroing_map_entry, zeroing_map_exit = state.add_map("zero_map", {
            "i": f"0:{N}",
            "j": f"0:{N}"
        },
                                                            schedule=dace.ScheduleType.Default)
        init_B = state.add_access("_b")
        zeroing_tasklet = state.add_tasklet("zeroing", {}, {"b_out"}, "b_out = 0")
        state.add_memlet_path(zeroing_map_entry, zeroing_tasklet, memlet=dace.Memlet())
        state.add_memlet_path(zeroing_tasklet,
                              zeroing_map_exit,
                              init_B,
                              src_conn="b_out",
                              memlet=dace.Memlet(f"_b[i,j]"))

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
        state.add_memlet_path(init_B,
                              comp_map_entry,
                              k_map_entry,
                              multiply_tasklet,
                              dst_conn="chol_row_k",
                              memlet=dace.Memlet("_b[row, k]"))
        state.add_memlet_path(init_B,
                              comp_map_entry,
                              k_map_entry,
                              multiply_tasklet,
                              dst_conn="chol_col_k",
                              memlet=dace.Memlet("_b[col, k]"))
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
            "write_out", {"A_row_col", "s_in", "B_col_col"}, {"B_out"}, """
if row == col:
    B_out = sqrt(A_row_col - s_in)
else:
    B_out = (A_row_col - s_in) / B_col_col
""")

        # inputs
        A_read = state.add_read("_a")
        state.add_memlet_path(s_access_out, write_out_tasklet, dst_conn="s_in", memlet=dace.Memlet(f"s[0]"))
        state.add_memlet_path(A_read,
                              comp_map_entry,
                              write_out_tasklet,
                              dst_conn="A_row_col",
                              memlet=dace.Memlet(f"_a[row, col]"))

        state.add_memlet_path(init_B,
                              comp_map_entry,
                              write_out_tasklet,
                              dst_conn="B_col_col",
                              memlet=dace.Memlet("_b[col, col]"))

        # output
        B_write = state.add_write("_b")
        state.add_memlet_path(write_out_tasklet,
                              comp_map_exit,
                              B_write,
                              src_conn="B_out",
                              memlet=dace.Memlet("_b[row, col]"))

        return sdfg
