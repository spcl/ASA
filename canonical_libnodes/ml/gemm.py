# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Canonical expansion for DaCeML GEMM node.

    This is mappend on an MMM canonical node, followed by an Addition library node to add
    the Bias to the multiplication result.
'''

import dace
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties
import copy
from canonical_libnodes.blas.mmm import MMM


@dace.library.expansion
class ExpandGemmAddC(ExpandTransformation):
    '''
    Adds C to the Gemm result
    '''
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        Y_no_bias = in_desc_with_name(node, parent_state, parent_sdfg, "Y_bn")
        Y_bias = out_desc_with_name(node, parent_state, parent_sdfg, "Y")
        C = in_desc_with_name(node, parent_state, parent_sdfg, "C")
        N = Y_bias.shape[0]
        M = Y_bias.shape[1]

        #### Add containers
        _, array_y_no_bias = sdfg.add_array("Y_bn", Y_no_bias.shape, dtype=Y_no_bias.dtype)
        _, array_y_bias = sdfg.add_array("Y", Y_bias.shape, dtype=Y_bias.dtype)
        _, array_c = sdfg.add_array("C", C.shape, dtype=C.dtype)

        y_no_bias_read = state.add_read("Y_bn")
        c_read = state.add_read("C")
        y_bias_write = state.add_write("Y")

        addmap_me, add_map_mx = state.add_map("gemm_add_map", {
            "n": f"0:{N}",
            "m": f"0:{M}",
        })
        tasklet = state.add_tasklet("add_c", {"in_y", "in_c"}, {"out_y"}, "out_y = in_y + in_c")

        # add memlets
        state.add_memlet_path(y_no_bias_read, addmap_me, tasklet, dst_conn="in_y", memlet=dace.Memlet("Y_bn[n,m]"))
        state.add_memlet_path(c_read, addmap_me, tasklet, dst_conn="in_c", memlet=dace.Memlet("C[m]"))
        state.add_memlet_path(tasklet, add_map_mx, y_bias_write, src_conn="out_y", memlet=dace.Memlet(f"Y[n,m]"))
        return sdfg


@dace.library.node
class GemmAddC(dace.sdfg.nodes.LibraryNode):
    '''
        Adds the C, row by row in the computed matrix
        
    '''

    # Global properties
    implementations = {"default": ExpandGemmAddC}
    default_implementation = "default"

    # Object fields
    beta = properties.SymbolicProperty(allow_none=False, default=1)

    def __init__(self, name, beta=1.0, location=None):
        super().__init__(name, location=location, inputs={"Y_bn", "C"}, outputs={"Y"})
        self.beta = beta

    def validate(self, sdfg, state):
        #TODO
        pass


@dace.library.expansion
class ExpandGemmCanonical(ExpandTransformation):
    ''' 
        Canonical Expansion for the GEMM ONNX operator.
        Uses MMM + an additional task for adding C (if needed)
    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        ### Checks

        A = in_desc_with_name(node, parent_state, parent_sdfg, "A")
        B = in_desc_with_name(node, parent_state, parent_sdfg, "B")
        C = in_desc_with_name(node, parent_state, parent_sdfg, "C")  # TODO: C may be optional
        Y = out_desc_with_name(node, parent_state, parent_sdfg, "Y")
        assert node.alpha == 1.0 and node.beta == 1.0
        assert node.transA == 0

        sdfg = dace.SDFG(node.label + "_sdfg")

        # Get descriptors (note this may have a shape that is not the one where we expect. E.g., a view on a higher order container
        # will show here the original shape of the container, not the viewed one. In that case use something like _get_mamul_operator)

        state = sdfg.add_state(node.label + "_state")
        sdfg.add_datadesc("A", copy.deepcopy(A))
        sdfg.add_datadesc("B", copy.deepcopy(B))
        sdfg.add_datadesc("C", copy.deepcopy(C))
        sdfg.add_datadesc("Y", copy.deepcopy(Y))
        sdfg.arrays["A"].transient = False
        sdfg.arrays["B"].transient = False
        sdfg.arrays["C"].transient = False
        sdfg.arrays["Y"].transient = False

        # GEMM Parameters
        N = A.shape[0]
        K = A.shape[1]
        M = B.shape[1]

        # Compute MatMul
        a_read = state.add_read("A")
        b_read = state.add_read("B")

        _, array_y_tmp = sdfg.add_array("Y_bn", Y.shape, dtype=Y.dtype, transient=True)

        y_access = state.add_access("Y_bn")

        libnode = MMM('MMM_gemm', transA=False, transB=bool(node.transB), alpha=1)

        libnode.implementation = "op_col"  # TODO: change it
        state.add_node(libnode)

        state.add_edge(a_read, None, libnode, '_a', dace.Memlet.from_array("A", sdfg.arrays["A"]))
        state.add_edge(b_read, None, libnode, '_b', dace.Memlet.from_array("B", sdfg.arrays["B"]))
        state.add_edge(libnode, '_c', y_access, None, dace.Memlet.from_array("Y_bn", array_y_tmp))

        # Add C
        # TODO: when we will move to consider Maps as Task, there will be no need to have an additional node
        c_read = state.add_read("C")
        y_write = state.add_write("Y")

        libnode = GemmAddC('GemmAddC')

        state.add_node(libnode)

        state.add_edge(y_access, None, libnode, 'Y_bn', dace.Memlet.from_array("Y_bn", array_y_tmp))
        state.add_edge(c_read, None, libnode, 'C', dace.Memlet.from_array("C", sdfg.arrays["C"]))
        state.add_edge(libnode, 'Y', y_write, None, dace.Memlet.from_array("Y", sdfg.arrays["Y"]))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandGemmCanonical.make_sdfg(node, state, sdfg)
