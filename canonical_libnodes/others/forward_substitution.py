# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Forward substitution library node and sequential, pure and canonical expansion
'''

import dace
from dace.transformation.transformation import ExpandTransformation
import copy
from dace import library


@dace.library.expansion
class ExpandForwardSubstitutionSeq(ExpandTransformation):
    '''
    Sequential, pure implementation of forward substitution

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
        assert N == M  # must be a squared matrix

        # Zeroing the result
        zeroing_map_entry, zeroing_map_exit = state.add_map("zero_map", {
            "i": f"0:{N}",
            "j": f"0:{N}"
        },
                                                            schedule=dace.ScheduleType.Default)
        init_B = state.add_access("_b")
        zeroing_tasklet = state.add_tasklet("zeroing", {}, {"_b_out"}, "_b_out = 0")
        state.add_memlet_path(zeroing_map_entry, zeroing_tasklet, memlet=dace.Memlet())
        state.add_memlet_path(zeroing_tasklet,
                              zeroing_map_exit,
                              init_B,
                              src_conn="_b_out",
                              memlet=dace.Memlet(f"_b[i,j]"))

        ## Computational maps (row and col)
        comp_map_entry, comp_map_exit = state.add_map("comp_map", {
            "row": f"0:{N}",
            "col": f"0:row+1"
        },
                                                      schedule=dace.ScheduleType.Sequential)

        # NOTE: for the sake of having something that is fully inlineable, I slightly changed the code
        # We compute in every case s, and then depending on the value of row and col we use it or not
        # The number of ops is the same

        ## init s
        sdfg.add_scalar("s", dtype, storage=dace.StorageType.Register, transient=True)
        s_access_in = state.add_access("s")
        s_access_out = state.add_access("s")

        init_s = state.add_tasklet("init_s", {}, {"s_out"}, "s_out = 0")

        state.add_memlet_path(comp_map_entry, init_s, memlet=dace.Memlet())
        state.add_memlet_path(init_s, s_access_in, src_conn="s_out", memlet=dace.Memlet(f"s[0]"))

        ## K map, accumulate over s
        # for k in range(col, row):
        # s += A[row, k] * fs[k, col]

        A_read = state.add_read("_a")

        k_map_entry, k_map_exit = state.add_map("k_map", {
            "k": f"col:row",
        }, schedule=dace.ScheduleType.Sequential)
        multiply_tasklet = state.add_tasklet("multiply", {"_a_row_k", "_b_k_col", "s_in"}, {"s_out"},
                                             "s_out = _a_row_k * _b_k_col + s_in")
        # inputs
        state.add_memlet_path(A_read,
                              comp_map_entry,
                              k_map_entry,
                              multiply_tasklet,
                              dst_conn="_a_row_k",
                              memlet=dace.Memlet("_a[row, k]"))
        state.add_memlet_path(init_B,
                              comp_map_entry,
                              k_map_entry,
                              multiply_tasklet,
                              dst_conn="_b_k_col",
                              memlet=dace.Memlet("_b[k, col]"))
        state.add_memlet_path(s_access_in, k_map_entry, multiply_tasklet, dst_conn="s_in", memlet=dace.Memlet(f"s[0]"))

        #output
        state.add_memlet_path(multiply_tasklet, k_map_exit, s_access_out, src_conn="s_out", memlet=dace.Memlet(f"s[0]"))

        ## Write tasklet
        # if row == col:
        #     fs[row, col] = 1 / A[row, col]
        # else:
        #     fs[row, col] = -s / A[row, row] #note, these two A are the same

        # TODO: if working with complex, the 1 must be casted with dace.complex64(1)
        write_out_tasklet = state.add_tasklet(
            "write_out", {"_a_row_row", "s_in"}, {"_b_out"}, """
if row == col:
    _b_out = 1/ _a_row_row
else:
    _b_out = -s_in / _a_row_row
""")

        # inputs
        state.add_memlet_path(s_access_out, write_out_tasklet, dst_conn="s_in", memlet=dace.Memlet(f"s[0]"))
        state.add_memlet_path(A_read,
                              comp_map_entry,
                              write_out_tasklet,
                              dst_conn="_a_row_row",
                              memlet=dace.Memlet(f"_a[row, row]"))

        # output
        _b_write = state.add_write("_b")
        state.add_memlet_path(write_out_tasklet,
                              comp_map_exit,
                              _b_write,
                              src_conn="_b_out",
                              memlet=dace.Memlet("_b[row, col]"))

        return sdfg


@dace.library.node
class ForwardSubstitution(dace.sdfg.nodes.LibraryNode):
    '''
        Perform TRMM by using forward substitution by solving the 
        system of systems AX=I

        A is of size NxN

        for row in range(N):
            for col in range(row + 1):
                s = 0
                if row == col:
                    X[row, col] = 1 / A[row, col]
                else:
                    for k in range(col, row):
                        s += A[row, k] * X[k, col]

                    X[row, col] = -s / A[row, row]
    '''

    # Global properties
    implementations = {"seq": ExpandForwardSubstitutionSeq}
    default_implementation = None

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, inputs={"_a"}, outputs={
            "_b",
        }, **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A two-tuple of the input and output descriptors
        """
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to pcholesky")
        in_memlet = in_edges[0].data
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one input from cholesky node")
        out_memlet = out_edges[0].data

        # Squeeze input memlets
        squeezed1 = copy.deepcopy(in_memlet.subset)
        sqdims1 = squeezed1.squeeze()
        # Squeeze output memlets
        squeezed2 = copy.deepcopy(out_memlet.subset)
        sqdims2 = squeezed2.squeeze()

        desc_ain, desc_aout, = None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_a":
                desc_ain = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_b":
                desc_bout = sdfg.arrays[e.data.data]

        if desc_ain.dtype.base_type != desc_bout.dtype.base_type:
            raise ValueError("Basetype of input and output must be equal!")

        stride_a = desc_ain.strides[sqdims1[0]]
        shape_a = squeezed1.size()
        rows_a = shape_a[0]
        cols_a = shape_a[1]
        stride_b = desc_bout.strides[sqdims2[0]]
        shape_b = squeezed2.size()
        rows_b = shape_b[0]
        cols_b = shape_b[1]

        if len(squeezed1.size()) != 2:
            print(str(squeezed1))
            raise ValueError("Forward Substitution only supported on 2-dimensional arrays")

        return desc_ain, shape_a, desc_bout, shape_b