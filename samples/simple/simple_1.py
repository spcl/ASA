# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    In this simple example, the user application consists of a simple
    Matrix-Matrix Multiplication.
    The examples illustrates the API to invoke the framework and the
    Design Space Exploration phase.

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
from dse.dse import DSE
from dse.analysis import get_pareto_configurations
import argparse


# The user application that will be considered for ASA construction
@dace.program
def simple_1(A: dace.float32[8, 8], B: dace.float32[8, 8]):
    return A @ B


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',
                        help='Execute the original program and validates the result',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    sdfg = simple_1.to_sdfg()

    if args.validate:
        np_dtype = np.float32

        # Initialize input
        A = np.random.rand(8, 8).astype(np.float32)
        B = np.random.rand(8, 8).astype(np.float32)
        C = sdfg(A, B)
        assert np.allclose(C, A @ B)

    # Perform Design Space Exploration
    results = DSE(
        sdfg,
        num_pes=[8, 16, 32, 64],  # List of allowed number of PEs 
        on_chip_memory_sizes=[32, 128]  # List of allowed on-chip-memory size
    )

    # Get pareto frontier configurations (results saved on pareto_simple_1_...csv)
    get_pareto_configurations(results, sdfg.name)
