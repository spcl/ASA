# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Let's assume for simplicity that everything is squared
    - x, y and z are N elements vector
    - A and B are NxN matrices

    The considered user application computes:
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

    The user application is written using the DaCe SDFG API (https://nbviewer.jupyter.org/github/spcl/dace/blob/master/tutorials/sdfg_api.ipynb)

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
from canonical_libnodes.blas.mmm import MMM
from canonical_libnodes.blas.op import OP
from canonical_libnodes.blas.mv import MV

from dse.dse import DSE
from dse.analysis import get_pareto_configurations


def create_sdfg(dtype, N, sdfg_name):

    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()
    A, A_arr = sdfg.add_array("A", (N, N), dtype)
    B, B_arr = sdfg.add_array("B", (N, N), dtype)
    C, C_arr = sdfg.add_array("C", (N, N), dtype)
    D, D_arr = sdfg.add_array("D", (N, N), dtype)
    E, E_arr = sdfg.add_array("E", (N, N), dtype)
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

    ### Final MV
    rz = state.add_read("z")
    ww = state.add_write("w")
    mv_libnode = MV('MV', transA=False, alpha=1, beta=0)

    state.add_node(mv_libnode)

    state.add_edge(wE, None, mv_libnode, '_A', dace.Memlet.from_array(E, E_arr))
    state.add_edge(rz, None, mv_libnode, '_x', dace.Memlet.from_array(z, z_arr))
    state.add_edge(mv_libnode, '_y', ww, None, dace.Memlet.from_array(w, w_arr))

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--validate',
                        help='Execute the original program and validates the result',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    validate = args.validate

    N = 8  # basic size of data

    # unique name for sdfg
    sdfg_name = f"simple_2_{N}"

    np_dtype = np.float32

    # Initialize arrays:
    A = np.random.rand(N, N).astype(np_dtype)
    B = np.random.rand(N, N).astype(np_dtype)
    C = np.random.rand(N, N).astype(np_dtype)
    D = np.random.rand(N, N).astype(np_dtype)
    E = np.random.rand(N, N).astype(np_dtype)
    x = np.random.rand(N).astype(np_dtype)
    y = np.random.rand(N).astype(np_dtype)
    z = np.random.rand(N).astype(np_dtype)
    w = np.random.rand(N).astype(np_dtype)

    sdfg = create_sdfg(dace.float32, N, sdfg_name)

    if validate:

        def ground_truth(A, B, x, y, z):
            D = A @ B
            C = np.outer(x, y)
            E = C @ D
            w = E @ z
            return w

        w_ref = ground_truth(A, B, x, y, z)
        sdfg(A=A, B=B, C=C, D=D, E=E, x=x, y=y, z=z, w=w)

        assert np.allclose(w, w_ref)

    # Perform Design Space Exploration
    results = DSE(sdfg, num_pes=[8, 16], on_chip_memory_sizes={64, 128})

    # Get pareto frontier configurations (results saved on pareto_simple_2_...csv)
    get_pareto_configurations(results, sdfg.name)
