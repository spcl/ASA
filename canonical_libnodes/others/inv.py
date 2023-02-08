# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Matrix Inversion Canonical expansions.
    Relies on: Cholesky, Forward Substitution, Transposition and Matrix-Matrix Multiplicaiton
"""

import dace
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace.transformation.transformation import ExpandTransformation
import copy
from canonical_libnodes.others.cholesky import ExpandCholeskySeq
from dace.libraries.linalg.nodes import Cholesky
from canonical_libnodes.others.forward_substitution import ForwardSubstitution
from dace.libraries.blas.nodes import Transpose
from canonical_libnodes.blas.mmm import MMM


@dace.library.expansion
class ExpandCubicCompute(ExpandTransformation):
    '''
    Mock expansions, used for the sake of analyzability.
    Use ExpandInvCholeskyFwdTRMM for a fully functional expansion.
    Expansion that results in a map with cubic number of iterations in N
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")

        Ain = in_desc_with_name(node, parent_state, parent_sdfg, "_ain")
        Aout = out_desc_with_name(node, parent_state, parent_sdfg, "_aout")

        ain_arr = sdfg.add_array('_ain', Ain.shape, dtype=Ain.dtype, strides=Ain.strides)
        aout_arr = sdfg.add_array('_aout', Aout.shape, dtype=Aout.dtype, strides=Aout.strides)

        assert Ain.shape == Aout.shape

        # Create a mock SDFG where the input is read 3 times, such that this seems to be a downsampler
        N, M = Ain.shape
        R = N  # how many reads

        ## Outer map loops over all output entries
        outer_me, outer_mx = state.add_map("output_map", {
            "i": f"0:{N}",
            "j": f"0:{M}",
        },
                                           schedule=dace.ScheduleType.Sequential)

        sdfg.add_scalar("value", Ain.dtype, storage=dace.StorageType.Register, transient=True)
        value_in = state.add_access("value")
        value_out = state.add_access("value")

        # this value is local in the map

        state.add_memlet_path(outer_me, value_in, memlet=dace.Memlet())

        ## Inner map to mimic multiple reads
        a_read = state.add_read("_ain")
        a_write = state.add_write("_aout")
        inner_me, inner_mx = state.add_map('inner_map', {
            "k": f"0:{R}",
        }, schedule=dace.ScheduleType.Sequential)

        read_tasklet = state.add_tasklet("read_tasklet",
                                         inputs={"a_in", "value_in"},
                                         outputs={"value_out"},
                                         code="value_out = a_in")  # value_in is not used

        state.add_memlet_path(value_in, inner_me, read_tasklet, dst_conn="value_in", memlet=dace.Memlet("value[0]"))
        state.add_memlet_path(a_read,
                              outer_me,
                              inner_me,
                              read_tasklet,
                              dst_conn="a_in",
                              memlet=dace.Memlet("_ain[i,j]"))
        state.add_memlet_path(read_tasklet, inner_mx, value_out, src_conn="value_out", memlet=dace.Memlet("value[0]"))
        state.add_memlet_path(value_out, outer_mx, a_write, memlet=dace.Memlet("_aout[i,j]"))
        return sdfg


@dace.library.node
class ComputeOverMatrix(dace.sdfg.nodes.LibraryNode):
    '''
        This is a dummy library node that can be used to implement dummy iteration over matrices.
        For example, it can be used to represent a cubic algorithms, that reads an input matrix N times
        and produces an output matrix.

        This will be represented as a downsampler canonical node
    '''

    # Global properties
    implementations = {
        "cubic": ExpandCubicCompute,
    }
    default_implementation = None

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs={"_ain"}, outputs={"_aout"})

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected 1 inputs to ComputeOverMatrix")

        in_memlet = in_edges[0].data
        in_subset = copy.deepcopy(in_memlet.subset)
        in_subset.squeeze()
        size_in = in_subset.size()

        if len(size_in) > 2:
            raise NotImplementedError("ComputeOverMatrix currently support 2D shapes")

        out_edges = state.out_edges(self)

        out_memlet = out_edges[0].data

        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        size_out = out_subset.size()
        if size_out != size_in:
            raise ValueError("Output shape of ComputeMatrix does not match with input shape.")


@dace.library.expansion
class ExpandInvDummy(ExpandTransformation):
    ''' 
        This is a dummy expansion for the inversion. This accounts for three N^3 map that should (roughly in term 
        of complexity) mimic the cholesky factorization, the forward substitution and the final triangular matrix multiplication
    
        Just wanted to validate the workflow
    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        Ain = in_desc_with_name(node, parent_state, parent_sdfg, "_ain")
        Aout = out_desc_with_name(node, parent_state, parent_sdfg, "_aout")

        ain, ain_arr = sdfg.add_array('_ain', Ain.shape, dtype=Ain.dtype, strides=Ain.strides)
        aout, aout_arr = sdfg.add_array('_aout', Aout.shape, dtype=Aout.dtype, strides=Aout.strides)

        assert Ain.shape == Aout.shape

        ### Cholesky
        a_fact, a_fact_arr = sdfg.add_array('a_fact_bn',
                                            Aout.shape,
                                            dtype=Aout.dtype,
                                            strides=Aout.strides,
                                            transient=True)
        rA = state.add_read('_ain')
        wA_fact = state.add_access('a_fact_bn')

        libnode = ComputeOverMatrix("Cholesky")
        libnode.implementation = "cubic"
        state.add_node(libnode)

        state.add_edge(rA, None, libnode, '_ain', dace.Memlet.from_array(ain, ain_arr))
        state.add_edge(libnode, '_aout', wA_fact, None, dace.Memlet.from_array(a_fact, a_fact_arr))

        ### Forward substitution
        a_fs, a_fs_arr = sdfg.add_array('a_fs_bn', Aout.shape, dtype=Aout.dtype, strides=Aout.strides, transient=True)
        wA_fs = state.add_access("a_fs_bn")
        libnode = ComputeOverMatrix("ForwardSubstitution")
        libnode.implementation = "cubic"
        state.add_node(libnode)

        state.add_edge(wA_fact, None, libnode, '_ain', dace.Memlet.from_array(a_fact, a_fact_arr))
        state.add_edge(libnode, '_aout', wA_fs, None, dace.Memlet.from_array(a_fs, a_fs_arr))

        ### Triangular matrix-matrix multiplication
        wA = state.add_access("_aout")
        libnode = ComputeOverMatrix("TRMM")
        libnode.implementation = "cubic"
        state.add_node(libnode)

        state.add_edge(wA_fs, None, libnode, '_ain', dace.Memlet.from_array(a_fs, a_fs_arr))
        state.add_edge(libnode, '_aout', wA, None, dace.Memlet.from_array(aout, aout_arr))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandInvDummy.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandInvCholeskyFwdTRMM(ExpandTransformation):
    ''' 
        This expansion expands INV as a combination of:
        - cholesky: A= LL^T
        - forward substitution: solve LU=I, where U=L^T A^-1. At the end U = L^-1
        - triangular matrix-matrix multiplication: A^-1 = (LL^T)^-1 = L^-T L^-1 = U^T U
    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        Ain = in_desc_with_name(node, parent_state, parent_sdfg, "_ain")
        Aout = out_desc_with_name(node, parent_state, parent_sdfg, "_aout")

        ain, ain_arr = sdfg.add_array('_ain', Ain.shape, dtype=Ain.dtype, strides=Ain.strides)
        aout, aout_arr = sdfg.add_array('_aout', Aout.shape, dtype=Aout.dtype, strides=Aout.strides)

        assert Ain.shape == Aout.shape

        ### Cholesky
        a_fact, a_fact_arr = sdfg.add_array('a_fact_bn',
                                            Aout.shape,
                                            dtype=Aout.dtype,
                                            strides=Aout.strides,
                                            transient=True)
        rA = state.add_read('_ain')
        wA_fact = state.add_access('a_fact_bn')

        cholesky_libnode = Cholesky("Cholesky")
        dace.libraries.linalg.nodes.cholesky.Cholesky.register_implementation("seq", ExpandCholeskySeq)
        cholesky_libnode.implementation = "seq"

        state.add_node(cholesky_libnode)

        state.add_edge(rA, None, cholesky_libnode, '_a', dace.Memlet.from_array(ain, ain_arr))
        state.add_edge(cholesky_libnode, '_b', wA_fact, None, dace.Memlet.from_array(a_fact, a_fact_arr))

        ### Forward substitution
        a_fs, a_fs_arr = sdfg.add_array('a_fs_bn', Aout.shape, dtype=Aout.dtype, strides=Aout.strides, transient=True)
        wA_fs = state.add_access("a_fs_bn")

        fwd_libnode = ForwardSubstitution('fs')

        fwd_libnode.implementation = "seq"
        state.add_node(fwd_libnode)

        state.add_edge(wA_fact, None, fwd_libnode, '_a', dace.Memlet.from_array(a_fact, a_fact_arr))
        state.add_edge(fwd_libnode, '_b', wA_fs, None, dace.Memlet.from_array(a_fs, a_fs_arr))

        ### Transposition
        # TODO.... conjugate
        # Note that we will be not able anyway to stream from forward substitution to TRMM given that uses both A and A transp
        a_fs_transp, a_fs_transp_arr = sdfg.add_array('a_fs_transp_bn',
                                                      Aout.shape,
                                                      dtype=Aout.dtype,
                                                      strides=Aout.strides,
                                                      transient=True)
        wA_fs_transp = state.add_access("a_fs_transp_bn")

        transpose_libnode = Transpose('transpose', dtype=Aout.dtype)
        transpose_libnode.implementation = "pure"

        state.add_edge(wA_fs, None, transpose_libnode, '_inp', dace.Memlet.from_array(a_fs, a_fs_arr))
        state.add_edge(transpose_libnode, '_out', wA_fs_transp, None,
                       dace.Memlet.from_array(a_fs_transp, a_fs_transp_arr))

        ### Triangular matrix-matrix multiplication
        # TODO: need a specific one for TRMM, right now I am using classical MMM

        wA = state.add_access("_aout")

        mmm_libnode = MMM('MMM', transA=False, transB=False, alpha=1)

        mmm_libnode.implementation = "mv"  # TODO: change it
        state.add_node(mmm_libnode)

        state.add_edge(wA_fs_transp, None, mmm_libnode, '_a', dace.Memlet.from_array(a_fs_transp, a_fs_transp_arr))
        state.add_edge(wA_fs, None, mmm_libnode, '_b', dace.Memlet.from_array(a_fs, a_fs_arr))
        state.add_edge(mmm_libnode, '_c', wA, None, dace.Memlet.from_array(aout, aout_arr))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandInvCholeskyFwdTRMM.make_sdfg(node, state, sdfg)
