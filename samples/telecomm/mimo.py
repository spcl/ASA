# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    In this sample, the application consist of a MIMO (Multiple-Input/Multiple-Output)
    detection, which is a common operation done in baseband communications.

    The algorithm is an iterative algorithm, and applies MIMO detection on
    different input signals.

    Note: currently the DSE does not output anything, but with default parameter
    this will require half an hour to complete on a 16 cores machine.
"""

import dace
import numpy as np
from dace.transformation.interstate import InlineSDFG
import argparse
from dse.dse import DSE
from dse.analysis import get_pareto_configurations
from canonical_libnodes.blas.add import AddConstant

# Number of iterations
N = dace.symbol('N', dace.int32)


@dace.program
def mimo(Ruu: dace.float64[N, 64, 64], H: dace.float64[N, 64, 8], R: dace.float64[N, 64, 1]):

    S = np.empty((N, 8, 1), dtype=np.float64)
    for i in dace.map[0:N]:
        Ruu_H = Ruu[i] @ H[i]
        local_H = H[i]
        HH_Ruu_H = np.transpose(local_H) @ Ruu_H
        AddConstant(inp=HH_Ruu_H, outp=HH_Ruu_H, constant=1)
        HH_Ruu_H_inv = np.linalg.inv(HH_Ruu_H)
        G = Ruu_H @ HH_Ruu_H_inv
        S[i] = np.transpose(G) @ R[i]
    return S


def ground_truth(Ruu, H, R):
    '''
        Ground Truth, used to validate the result
    '''
    Ruu_H = Ruu @ H
    HH_Ruu_H = np.transpose(H) @ Ruu_H + 1
    HH_Ruu_H_inv = np.linalg.inv(HH_Ruu_H)
    G = Ruu_H @ HH_Ruu_H_inv
    S = np.transpose(G) @ R
    return S


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-N",
                        type=int,
                        nargs="?",
                        default=2,
                        help="Number of input signals (2 by default, 32760 typical)")
    parser.add_argument("-U", type=int, nargs="?", default=1, help="Unrolling factor (1 by default, 36 typical)")
    parser.add_argument('--validate',
                        help='Execute the original program and validates the result',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    n = args.N
    unroll = args.U
    validate = args.validate

    print(f"Running MIMO with {n} input signals. DSE performed with unrolling factor {unroll}.")

    ## Generate input data
    inp = np.random.rand(n, 64, 64).astype(np.float64)
    Ruu = np.zeros((n, 64, 64), dtype=np.float64)
    for i in range(n):
        Ruu[i] = inp[i].T @ inp[i]

    H = np.random.rand(n, 64, 8).astype(np.float64)
    R = np.random.rand(n, 64, 1).astype(np.float64)

    ## Get SDFG
    sdfg = mimo.to_sdfg(simplify=True)
    sdfg.replace("N", n)  # Specialize application: this is needed for static analysis

    results = DSE(
        sdfg,
        num_pes=[32, 64, 128, 256, 512, 1024],
        on_chip_memory_sizes=[256, 1024, 8192],
        unroll_factor=unroll,
        use_multithreading=True,  # use multithreading
        n_threads=16,
    )

    # Get pareto frontier configurations
    get_pareto_configurations(results, sdfg.name)

    if validate:
        # compute ground truth
        S_ref = np.zeros((n, 8, 1), dtype=np.float64)
        for i in range(n):
            S_ref[i] = ground_truth(Ruu[i], H[i], R[i])

        S = sdfg(Ruu=Ruu, H=H, R=R, N=n)

        # Check
        for i in range(n):
            diff = np.linalg.norm(S[i] - S_ref[i]) / np.linalg.norm(S_ref[i])

            assert np.allclose(S[i], S_ref[i])
