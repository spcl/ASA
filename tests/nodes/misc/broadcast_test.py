# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
   Tests for broadcast library node
"""

from canonical_libnodes.misc.broadcast import Broadcast
import dace
import numpy as np
import pytest
from dace.transformation.transformation import ExpandTransformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG


def create_broadcast_sdfg(dtype, in_shape, out_shape, row_major, implementation, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    inp, inp_arr = sdfg.add_array("inp", in_shape, dtype)
    outp, outp_arr = sdfg.add_array("outp", out_shape, dtype)

    rin = state.add_read("inp")
    wout = state.add_write("outp")
    libnode = Broadcast('Broadcast', row_major=row_major)

    libnode.implementation = implementation
    state.add_node(libnode)

    state.add_edge(rin, None, libnode, '_in', dace.Memlet.from_array(inp, inp_arr))
    state.add_edge(libnode, '_out', wout, None, dace.Memlet.from_array(outp, outp_arr))

    return sdfg


def run_test(implementation="broadcast", row_major=True, N=4, M=5):

    # Currently no support for alpha, trans or beta
    # unique name for sdfg
    sdfg_name = f"{implementation}_{N}_{M}"

    # shape of the transposed arrays
    in_shape = [N, M]

    ## Transp not currently supported
    print(f'Broadcast {N}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    inp = np.random.rand(*in_shape).astype(np_dtype)
    outp = np.random.rand(*in_shape).astype(np_dtype)

    regression = np.copy(inp)

    sdfg = create_broadcast_sdfg(dace.float32, in_shape, in_shape, row_major, implementation, sdfg_name)

    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    sdfg(inp=inp, outp=outp)

    # TODO test row/col major
    # We don't have a way of checking the actual access
    assert np.allclose(outp, regression)


@pytest.mark.parametrize("row_major", [True, False])
def test_broadcast(row_major):
    run_test(row_major=row_major)


if __name__ == "__main__":
    run_test(row_major=True)
    run_test(row_major=False)
