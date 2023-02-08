# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Test Add node with replacement
'''

import dace
import numpy as np
from canonical_libnodes.blas.add import AddConstant
from dace.transformation.interstate import InlineSDFG

N, M = (dace.symbol(s, dtype=dace.int32) for s in ('N', 'M'))


@dace.program
def add_test(A: dace.float32[N, M], B: dace.float32[N, M]):
    # B[:] = A + 1
    AddConstant(inp=A, outp=B, constant=1)


def test_add(implementation="pure", N=4, M=5):

    # shape of the transposed arrays
    in_shape = [N, M]

    ## Transp not currently supported
    print(f'Add {N}x{M}')

    np_dtype = np.float32

    # Initialize arrays: Randomize A and B, zero C
    inp = np.random.rand(*in_shape).astype(np_dtype)
    outp = np.random.rand(*in_shape).astype(np_dtype)

    regression = inp + 1

    sdfg = add_test.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    # sdfg.view()

    sdfg(A=inp, B=outp, N=N, M=M)

    assert np.allclose(outp, regression)


if __name__ == "__main__":
    test_add()
