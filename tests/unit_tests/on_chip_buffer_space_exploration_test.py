"""
    This suites contains various tests for on-chip buffer space exploration
    
    We run DSE for various on-chip buffer space budges and we check
    the results.
    
    TODO: add more tests
"""

import pytest
import dace
import numpy as np
from dace.transformation.transformation import ExpandTransformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG
from canonical_libnodes.blas.mmm import MMM
from canonical_libnodes.blas.op import OP
from canonical_libnodes.blas.mv import MV
from dse.dse import DSE


def test_sample_three():
    """
        Computes:
        D = A @ B
        C = np.outer(u, v)
        E = C @ D
        w = E @ z

            x    y         A    B
            │    │         │    │
            │    │         │    │
            ┌▼────▼─┐      ┌▼────▼─┐
            │   OP  │      │  MMM  │
            └───┬───┘      └───┬───┘
                │ C            │ D
                │          ┌───▼───┐
                └──────────►  MMM  │
                            └───┬───┘   z
                                │  E    │
                            ┌───▼───┐   │
                            │  MV   ◄───┘
                            └──┬────┘
                                ▼  w
    """
    def create_sdfg(dtype, N, sdfg_name, mv_implementation="seq", op_implementation="OP_by_col"):

        sdfg = dace.SDFG(sdfg_name)
        state = sdfg.add_state()
        A, A_arr = sdfg.add_array("A", (N, N), dtype)
        B, B_arr = sdfg.add_array("B", (N, N), dtype)
        C, C_arr = sdfg.add_array("C", (N, N), dtype)  # in theory it can be transient
        D, D_arr = sdfg.add_array("D", (N, N), dtype)  # in theory it can be transient
        E, E_arr = sdfg.add_array("E", (N, N), dtype)  # in theory it can be transient
        x, x_arr = sdfg.add_array("x", (N, ), dtype)
        y, y_arr = sdfg.add_array("y", (N, ), dtype)
        z, z_arr = sdfg.add_array("z", (N, ), dtype)
        w, w_arr = sdfg.add_array("w", (N, ), dtype)

        ### First Matrix multiplication

        rA = state.add_read("A")
        rB = state.add_read("B")
        wD = state.add_access("D")

        mmm_libnode_1 = MMM('MMM')

        state.add_node(mmm_libnode_1)

        state.add_edge(rA, None, mmm_libnode_1, '_a', dace.Memlet.from_array(A, A_arr))
        state.add_edge(rB, None, mmm_libnode_1, '_b', dace.Memlet.from_array(B, B_arr))
        state.add_edge(mmm_libnode_1, '_c', wD, None, dace.Memlet.from_array(D, D_arr))

        ### Outer product

        rx = state.add_read("x")
        ry = state.add_read("y")
        wC = state.add_access("C")
        op_libnode = OP('OP')
        op_libnode.implementation = op_implementation
        state.add_node(op_libnode)

        state.add_edge(rx, None, op_libnode, '_u', dace.Memlet.from_array(x, x_arr))
        state.add_edge(ry, None, op_libnode, '_v', dace.Memlet.from_array(y, y_arr))
        state.add_edge(op_libnode, '_A', wC, None, dace.Memlet.from_array(C, C_arr))

        ### Second MMM (computes E)

        wE = state.add_access("E")

        mmm_libnode_2 = MMM('MMM2')

        state.add_node(mmm_libnode_2)

        state.add_edge(wC, None, mmm_libnode_2, '_a', dace.Memlet.from_array(C, C_arr))
        state.add_edge(wD, None, mmm_libnode_2, '_b', dace.Memlet.from_array(D, D_arr))
        state.add_edge(mmm_libnode_2, '_c', wE, None, dace.Memlet.from_array(E, E_arr))

        ### MV
        rz = state.add_read("z")
        ww = state.add_write("w")
        mv_libnode = MV('MV', transA=False, alpha=1, beta=0)
        mv_libnode.implementation = mv_implementation
        state.add_node(mv_libnode)

        state.add_edge(wE, None, mv_libnode, '_A', dace.Memlet.from_array(E, E_arr))
        state.add_edge(rz, None, mv_libnode, '_x', dace.Memlet.from_array(z, z_arr))
        state.add_edge(mv_libnode, '_y', ww, None, dace.Memlet.from_array(w, w_arr))

        return sdfg

    N = 4
    sdfg_name = f"unit_test_sample_three"
    np_dtype = np.float32
    sdfg = create_sdfg(dace.float32, N, sdfg_name)

    results = DSE(sdfg, num_pes=[8, 16], use_multithreading=True, on_chip_memory_sizes={32, 128})

    ## check some of the solutions
    # TODO: more robust checks

    # 8 PEs
    results_8 = [x for x in results[8]]

    # the first two should be the most performing ones with the same expansion list, different buffer space
    assert results_8[0].expansion_list == results_8[1].expansion_list
    assert results_8[0].on_chip_space < results_8[1].on_chip_space
    assert results_8[0].on_chip_IOs == 224
    assert results_8[1].on_chip_IOs == 388
    # this checks can be also less strict, something like
    assert results_8[0].off_chip_IOs > results_8[1].off_chip_IOs

    # 16 PEs
    results_16 = [x for x in results[16]]

    # the first two should be the most performing ones with the same expansion list, different buffer space
    assert results_16[0].expansion_list == results_16[1].expansion_list
    assert results_16[0].on_chip_space < results_16[1].on_chip_space
    assert results_16[0].on_chip_IOs == 160
    assert results_16[1].on_chip_IOs == 400
    # this checks can be also less strict, something like
    assert results_16[0].off_chip_IOs > results_16[1].off_chip_IOs
