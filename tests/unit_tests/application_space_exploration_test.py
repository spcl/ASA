"""
    This suites contains various small tests
    for application space exploration.
    Each of them runs the converter and verifies that the 
    exploration gives the expected results (in terms of makespan).


    Note: these tests do not check that the application semantic is correct
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
    npes = 16
    results = DSE(sdfg, num_pes=npes, use_multithreading=True)
    makespans = [x.makespan for x in results[npes]]

    expected = [35, 36, 36, 49, 49, 50, 51, 51, 52]
    assert makespans == expected


def test_special_in_volume_nodes():
    """
        Test for nodes that require special handling in computing the input volume
        (e.g., Reduce)
    """

    # Actually, this is already tested with the sample_three
    pass


def test_daceml():
    """
        Test with a DaCeML small model (conv+relu+maxpool)
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from daceml.torch import DaceModule, dace_module
    from dace.transformation.dataflow import RedundantArray
    import daceml

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            #first conv
            self.conv = nn.Conv2d(1, 6, 5)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv(x)), 2)
            return x

    # create pytorch model
    ptmodel = Model()

    #create data
    x = torch.rand((1, 1, 16, 16))

    impl = "pure"
    with dace.library.change_default(daceml.onnx.nodes.onnx_op.ONNXMaxPool,
                                     impl), dace.library.change_default(daceml.onnx.nodes.onnx_op.ONNXConv, impl):
        dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)

    sdfg = dace_model.sdfg
    sdfg.apply_transformations_repeated([RedundantArray], print_report=True)

    npes = 128
    results = DSE(sdfg, num_pes=npes)
    makespans = [x.makespan for x in results[npes]]

    expected = [6195, 7059, 8933]

    assert expected == makespans


def test_iterative_simple():
    """
    Tests for simple iterative computations
    """

    ### Just a single MV

    N = dace.symbol('N', dace.int32)

    @dace.program
    def iterated_mv(A: dace.float32[N, 16, 8], x: dace.float32[N, 8]):
        y = np.empty((N, 16), dtype=np.float32)
        for ii in dace.map[0:N]:
            y[ii] = A[ii] @ x[ii]

        return y

    n = 8
    A = np.random.rand(n, 16, 8).astype(np.float32)
    x = np.random.rand(n, 8).astype(np.float32)

    sdfg = iterated_mv.to_sdfg()
    sdfg.replace("N", n)

    npes = 4
    results = DSE(sdfg, num_pes=npes, use_multithreading=False, unroll_factor=1)
    makespans = [x.makespan for x in results[npes]]

    expected = [1024]

    assert expected == makespans

    # Increase the unroll factor
    results = DSE(sdfg, num_pes=npes, use_multithreading=False, unroll_factor=4)
    makespans = [x.makespan for x in results[npes]]

    expected = [256]
    assert expected == makespans

    ### A MMM followed by a MV

    @dace.program
    def iterated_mmm_mv(A: dace.float32[N, 8, 4], B: dace.float32[N, 4, 4], x: dace.float32[N, 4]):
        y = np.empty((N, 8), dtype=np.float32)
        for ii in dace.map[0:N]:
            y[ii] = (A[ii] @ B[ii]) @ x[ii]

        return y

    n = 8
    A = np.random.rand(n, 8, 4).astype(np.float32)
    B = np.random.rand(n, 4, 4).astype(np.float32)
    x = np.random.rand(n, 4).astype(np.float32)

    sdfg = iterated_mmm_mv.to_sdfg()
    sdfg.replace("N", n)

    npes = 16
    results = DSE(sdfg, num_pes=npes, use_multithreading=False, unroll_factor=1)
    makespans = [x.makespan for x in results[npes]]

    expected = [272, 280, 280]
    assert expected == makespans

    ## Increasing the unroll factor we halve the execution time for some of them
    results = DSE(sdfg, num_pes=npes, use_multithreading=False, unroll_factor=2)
    makespans = [x.makespan for x in results[npes]]

    expected = [136, 140, 276]
    assert expected == makespans
