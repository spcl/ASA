# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    In this sample, we show how to deal with iterated computations,
    where the computation is repeated multiple times but on different input data.
    

    In this case we (forcely) compute a matrix multiplication followed by
    a matrix-vector multiplication: 

        for i in range(N):
            y[i] = (A[i] @ B[i]) @ x[i]
    
    where A and B are NxN matrices, and x and y are N elements vectors.

    Note that each for loop iteration is independent from the others.

    In this case, assuming N is known at analysis, we could fully unroll the for-loop and work on the computation.
    However, if N is large, this can result in a very large SDFG, with the same computation (expressed by the loop body)
    repeated multiple times. To save DSE time, we allow the user to _partially_ unroll the computation.
    In this way, the resulting SDFG will be smaller, but we can still exploit the parallelism due to the independent sub-computations.



    By default, the program executes the Design Space Exploration and prints the 
    pareto frontier solutions.

    The details of all explored solutions are stored in a set of csv files for 
    successive analysis. For each considered number of processing elements,
    the corresponding csv file will contain:
    - the makespan (the expected running time) of the configuration 
    - the number of off-chip and on-chip IOs
    - the number of streaming IOs (the data movements volume occurring using on-chip inter-PE communications)
    - the various Power/Performance/Area scores
    - other statistics that can enable further analysis.

    We recommend to invoke the script with the DaCe debugprint configuration option disabled to avoid
    unnecessary messages.
"""

import dace
import numpy as np
import argparse
from canonical_libnodes.blas.mv import ExpandMVSeq
from dse.dse import DSE

N = dace.symbol('N', dace.int32)


@dace.program
def iterated_mmm_mv(A: dace.float32[N, 8, 4], B: dace.float32[N, 4, 4], x: dace.float32[N, 4]):
    y = np.empty((N, 8), dtype=np.float32)
    for ii in dace.map[0:N]:
        y[ii] = (A[ii] @ B[ii]) @ x[ii]
    return y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',
                        help='Execute the original program and validates the result',
                        action='store_true',
                        default=False)
    parser.add_argument("-N", type=int, nargs="?", default=8, help="Number of iterations (8 by default)")
    parser.add_argument("-U", type=int, nargs="?", default=2, help="Unrolling factor (2 by default)")
    args = parser.parse_args()
    n = args.N
    u = args.U
    validate = args.validate

    A = np.random.rand(n, 8, 4).astype(np.float32)
    B = np.random.rand(n, 4, 4).astype(np.float32)
    x = np.random.rand(n, 4).astype(np.float32)

    sdfg = iterated_mmm_mv.to_sdfg()
    sdfg.replace("N", n)

    if validate:
        y_ref = np.empty((n, 8), dtype=np.float32)
        for i in range(n):
            y_ref[i] = (A[i] @ B[i]) @ x[i]

        dace.libraries.blas.Gemv.register_implementation("seq", ExpandMVSeq)
        dace.libraries.blas.Gemv.default_implementation = "seq"

        y = sdfg(A=A, B=B, x=x, N=n)

        assert np.allclose(y, y_ref)

    # Perform DSE. The results will be stored in CSV file, whose names indicate
    # the number of iterations and the unrolling factor
    DSE(sdfg, num_pes=16, unroll_factor=u)
