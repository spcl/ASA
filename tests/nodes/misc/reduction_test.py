# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
   Tests for reduction library node
"""

from canonical_libnodes.misc.reduction import ReduceMMM
import dace
import numpy as np
import pytest
from dace.transformation.interstate import InlineSDFG


def create_reduction_sdfg(dtype, in_shape, out_shape, row_major, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    inp, inp_arr = sdfg.add_array("inp", in_shape, dtype)
    outp, outp_arr = sdfg.add_array("outp", out_shape, dtype)

    rin = state.add_read("inp")
    wout = state.add_write("outp")
    libnode = ReduceMMM('Reduce', row_major=row_major)

    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rin, None, libnode, '_in', dace.Memlet.from_array(inp, inp_arr))
    state.add_edge(libnode, '_out', wout, None, dace.Memlet.from_array(outp, outp_arr))

    return sdfg


def run_test(implementation="reduce_sum_col", K=3, N=4, M=5):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{N}_{M}"

    # shape of the transposed arrays
    in_shape = [K, N, M]
    out_shape = [N, M]
    ## Transp not currently supported
    print(f'Reduce {K}x{N}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    inp = np.random.rand(*in_shape).astype(np_dtype)
    outp = np.random.rand(*out_shape).astype(np_dtype)

    regression = inp.sum(axis=0)

    sdfg = create_reduction_sdfg(dace.float32, in_shape, out_shape, True, implementation, sdfg_name)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    sdfg(inp=inp, outp=outp)

    assert np.allclose(outp, regression)


def test_reduction():
    run_test()


if __name__ == "__main__":
    run_test()