# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Canonical expansion for Matmul ONNX node
'''

import dace
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties
import copy
import daceml
from canonical_libnodes.blas.mmm import MMM


@dace.library.expansion
class ExpandMatmulCanonical(ExpandTransformation):
    ''' 
        Canonical Expansion for the Matmul ONNX operator.
        Uses MMM

        NOTE:
        - for the sake of simplicity, we flattned A (which usually have 3 dimensions BxNxK -> BNxK), and then we properly deal with B in MMM (B is 2 or 3D)
            The result Y is still flattened
        - In this way we limit the parallelism to "M" or "K"
        - the LMV expansion is a bit different: this by default parallelizes along N, which is BxN so this can produce very large multiplications
    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        ### Checks

        A = in_desc_with_name(node, parent_state, parent_sdfg, "A")
        B = in_desc_with_name(node, parent_state, parent_sdfg, "B")
        Y = out_desc_with_name(node, parent_state, parent_sdfg, "Y")

        sdfg = dace.SDFG(node.label + "_sdfg")

        # Get descriptors (note this may have a shape that is not the one where we expect. E.g., a view on a higher order container
        # will show here the original shape of the container, not the viewed one. In that case use something like _get_mamul_operator)

        state = sdfg.add_state(node.label + "_state")
        sdfg.add_datadesc("A", copy.deepcopy(A))
        sdfg.add_datadesc("B", copy.deepcopy(B))
        sdfg.add_datadesc("Y", copy.deepcopy(Y))
        sdfg.arrays["A"].transient = False
        sdfg.arrays["B"].transient = False
        sdfg.arrays["Y"].transient = False

        # Matmul Parameters
        N = Y.shape[-2]
        M = Y.shape[-1]
        K = A.shape[-1]

        assert len(A.shape) <= 3
        assert len(B.shape) <= 3

        # Compute MatMul
        a_read = state.add_read("A")
        b_read = state.add_read("B")
        y_write = state.add_write("Y")

        ## This is a batched matmul, we need to flat (and treat as transients) the inputs and the output
        ## if needed

        if len(A.shape) > 2:
            # flatten A in this case
            NA = A.shape[0] * A.shape[1]
            _, array_a_flattened = sdfg.add_array("A_flat", [NA, A.shape[2]], dtype=A.dtype, transient=True)
            a_flat_access = state.add_access("A_flat")
            state.add_edge(a_read, None, a_flat_access, None, dace.Memlet.from_array('A', sdfg.arrays["A"]))

        if len(Y.shape) > 2:
            # flatten
            NY = Y.shape[0] * Y.shape[1]
            _, array_y_flattened = sdfg.add_array("Y_flat", [NY, Y.shape[2]], dtype=Y.dtype, transient=True)
            y_flat_access = state.add_access("Y_flat")
            state.add_edge(y_flat_access, None, y_write, None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))

        libnode = MMM('MMM_gemm', transA=False, transB=False, alpha=1, batched=len(B.shape) > 2)

        libnode.implementation = "LMV_col"  # TODO: support lmv

        state.add_node(libnode)

        if len(A.shape) > 2:
            state.add_edge(a_flat_access, None, libnode, '_a', dace.Memlet.from_array("A_flat", array_a_flattened))
        else:
            state.add_edge(a_read, None, libnode, '_a', dace.Memlet.from_array("A", sdfg.arrays["A"]))

        # if len(B.shape) > 2:
        #     state.add_edge(b_flat_access, None, libnode, '_b', dace.Memlet.from_array("B_flat", array_b_flattened))
        # else:
        state.add_edge(b_read, None, libnode, '_b', dace.Memlet.from_array("B", sdfg.arrays["B"]))
        if len(Y.shape) > 2:
            state.add_edge(libnode, '_c', y_flat_access, None, dace.Memlet.from_array("Y_flat", array_y_flattened))
        else:
            state.add_edge(libnode, '_c', y_write, None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))
        # import pdb
        # pdb.set_trace()
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandMatmulCanonical.make_sdfg(node, state, sdfg)
