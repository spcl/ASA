# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.

from canonical_libnodes.blas import mmm
from canonical_libnodes.misc import reduction
from canonical_libnodes.others import forward_substitution, cholesky
from dace.libraries import linalg
import dace
import networkx as nx
from collections import defaultdict
from typing import Dict, Set
from dace.transformation.dataflow import StreamingComposition
from dace.transformation.interstate import InlineSDFG
from canonical_libnodes import *
import copy
import time
from datetime import timedelta
from dse.utils import *
from dse.data import Result
import numpy as np
import statistics

# use passes instead of InlineSDFG
from dace.transformation.passes.fusion_inline import InlineSDFGs

from dse.analysis import ppa_model_base

# DaCeML (We need to activate this only if needed)
use_daceml = True
try:
    from daceml.onnx.nodes import onnx_op
    import daceml.onnx as donnx
except ImportError:
    print("Not using DaCeML")
    use_daceml = False

# Multi-threading
from multiprocessing import Queue, Process, Manager

## Scheduling import
from streamingsched.sched import StreamingScheduler, spatial_block_partitioning, eft
from streamingsched.utils.metrics import makespan
from streamingsched.sched.utils import build_W_matrix_HEFT
from streamingsched.sched.utils import print_schedule as print_sched
from streamingsched.utils.visualize import visualize_dag


def is_buffer_node(node: dace.nodes.AccessNode, graph, fully_expanded_sdfg):
    '''
    Understand if an access node must be a buffer node or not. 
    Buffer nodes are needed when data must be stored for reshaping, accumulation, reordering, replication, ...
    Otherwise the framework tries to stream data between tasks.

    We distinguish between the following cases:
    
    - if the node is a transient: we leniently consider transient access nodes as non buffer nodes,
       and we currently perform basic checks to verify that this is really the case (TODO: make this
       more robust)

    - if the access node is not a transient:
        - check if it connected to a map entry and map exit in the fully expanded SDFG
        - if no -> this is a buffer (safest option, maybe it can be improved)
        - if yes -> check that streaming composition can be applied
            - if yes: this is not a buffer
            - otherwise, this is a buffer 

    - explicitly defined buffer node: in every case, the user may explicitly require to consider
        an access node as buffer nodes by appending the "_bn" prefix to its name

    @return True if the given access node is a buffer node, false otherwise
    '''

    # TODO: deal properly with nodes that we know are anyway buffer nodes

    if node.data.endswith("_bn"):
        return True  # this is coming from the expansion

    # TODO: view can also not be buffer nodes (see mimo sample). We can thinking of optimizing this
    if isinstance(node.desc(graph), dace.data.View):
        return True

    nodedesc = node.desc(graph)
    if not nodedesc.transient and (graph.in_degree(node) != 1 or graph.out_degree(node) != 1):
        # This must be a buffer node (improve)
        return True

    # Now look for this specific node in the fully expanded SDFG
    # get the predecessor in the fully expanded_sdfg
    expanded_node = find_access_node_by_name(node.label, fully_expanded_sdfg)
    expanded_state = fully_expanded_sdfg.nodes()[0]  #

    # if this is a transient, check that the sum of input volume == sum of output volume: if is this
    # is the case, this is not a buffer node (e.g., partial_sum in reduction), otherwise yes (e.g., input A
    # in a MMM with OP expansion, that requires anyway buffering)

    if nodedesc.transient:

        # Corner case: this is something like a broadcast or a gather:
        # It has a single input or output element, and the input volume (on one edge) is equal to the output
        # volume on all edges.
        # TODO: Not 100% sure about this: this does not takes into account the actual access pattern of data
        # We should apply streaming composition also here, especially for certain cases (see later on)
        # or be sure that the application uses correctly transients
        if graph.in_degree(node) == 1 and graph.out_degree(node) > 1:
            # broadcast
            in_volume = int(expanded_state.in_edges(expanded_node)[0].data.volume)

            for out_edge in expanded_state.out_edges(expanded_node):
                if int(out_edge.data.volume) != in_volume:
                    break
            else:
                return False
        elif graph.in_degree(node) > 1 and graph.out_degree(node) == 1:
            # gather
            out_volume = int(expanded_state.out_edges(expanded_node)[0].data.volume)

            for in_edge in expanded_state.in_edges(expanded_node):
                if int(in_edge.data.volume) != out_volume:
                    break
            else:
                return False

        sum_input_volume = 0
        sum_output_volume = 0
        for in_edge in expanded_state.in_edges(expanded_node):
            sum_input_volume += int(in_edge.data.volume)

        for out_edge in expanded_state.out_edges(expanded_node):
            sum_output_volume += int(out_edge.data.volume)

        # Potential situation where we should check for streaming composition
        # if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
        #     in_edge = expanded_state.in_edges(expanded_node)[0]
        #     if not isinstance(in_edge.src, dace.nodes.MapExit):
        #         return sum_output_volume != sum_input_volume

        #     out_edge = expanded_state.out_edges(expanded_node)[0]
        #     if not isinstance(out_edge.dst, dace.nodes.MapEntry):
        #         return sum_output_volume != sum_input_volume
        #     xf = StreamingComposition()
        #     xf.first = in_edge.src
        #     xf.access = expanded_node
        #     xf.second = out_edge.dst
        #     can_be_applied = xf.can_be_applied(expanded_state, 0, fully_expanded_sdfg)

        #     return not can_be_applied
        return sum_output_volume != sum_input_volume

    in_edge = expanded_state.in_edges(expanded_node)[0]
    if not isinstance(in_edge.src, dace.nodes.MapExit):
        return True

    out_edge = expanded_state.out_edges(expanded_node)[0]
    if not isinstance(out_edge.dst, dace.nodes.MapEntry):
        return True

    # check if Streaming Composition can be applied

    # TODO: deal with the case of multiple output edges: if they all have the same volume
    #     this can be streamed as well. This must be handled directly in the Streaming
    #       composition transformation

    xf = StreamingComposition()

    xf.first = in_edge.src
    xf.access = expanded_node
    xf.second = out_edge.dst
    can_be_applied = xf.can_be_applied(expanded_state, 0, fully_expanded_sdfg)

    # if not can_be_applied:
    #     print(f"StreamingComposition can not be applied to {expanded_node}")
    # else:
    #     print(f"StreamingComposition, node {expanded_node} can be a stream")
    return not can_be_applied


def specialize_matmul(sdfg: dace.SDFG):
    '''
    Given a one-level SDFG, specializes all the matmul nodes (by expanding them)
    '''
    assert len(sdfg.nodes()) == 1
    state = sdfg.nodes()[0]
    for node in state.nodes():
        if isinstance(node, dace.libraries.blas.MatMul):
            node.expand(sdfg, state)


def register_canonical_implementations():
    '''
        Registers all the canonical implementation on the related libnodes
        TODO: do it in a more programmatic way
    '''

    # Classical nodes
    from canonical_libnodes.blas.mmm import ExpandMMM_MV, ExpandMMM_LMV, ExpandMMM_OP_Col
    dace.libraries.blas.Gemm.register_implementation("canonical_mv", ExpandMMM_MV)
    dace.libraries.blas.Gemm.register_implementation("canonical_lmv", ExpandMMM_LMV)
    dace.libraries.blas.Gemm.register_implementation("canonical_op_col", ExpandMMM_OP_Col)

    # Gemv
    from canonical_libnodes.blas.mv import ExpandMVSeq
    dace.libraries.blas.Gemv.register_implementation("canonical", ExpandMVSeq)

    if hasattr(dace.libraries, 'linalg'):
        from canonical_libnodes.others.inv import ExpandInvDummy, ExpandInvCholeskyFwdTRMM
        # dace.libraries.linalg.Inv.register_implementation("canonical_dummy",
        #                                                   ExpandInvDummy)  #so, this is not really canonical
        dace.libraries.linalg.Inv.register_implementation("canonical",
                                                          ExpandInvCholeskyFwdTRMM)  #so, this is not really canonical
        dace.libraries.linalg.Inv.register_implementation("canonical", ExpandInvCholeskyFwdTRMM)

        dace.libraries.linalg.Cholesky.register_implementation("seq", cholesky.ExpandCholeskySeq)
    # ML nodes
    if use_daceml:
        from canonical_libnodes.ml.im2col_conv import ExpandConvCanonical
        from canonical_libnodes.ml.gemm import ExpandGemmCanonical
        from canonical_libnodes.ml.maxpool import ExpandMaxPoolCanonical
        from canonical_libnodes.ml.softmax import ExpandSoftmaxCanonical
        from canonical_libnodes.ml.matmul import ExpandMatmulCanonical
        donnx.nodes.onnx_op.ONNXConv.register_implementation("canonical", ExpandConvCanonical)
        donnx.nodes.onnx_op.ONNXMaxPool.register_implementation("canonical", ExpandMaxPoolCanonical)
        donnx.nodes.onnx_op.ONNXGemm.register_implementation("canonical", ExpandGemmCanonical)
        donnx.nodes.onnx_op.ONNXSoftmax.register_implementation("canonical", ExpandSoftmaxCanonical)
        donnx.nodes.onnx_op.ONNXMatMul.register_implementation("canonical", ExpandMatmulCanonical)


def set_node_expansions(sdfg):
    '''
        For certain nodes that can be expanded directly to canonical form, we can not apply all expansions,
        but only those who result in canonical ones.
        

        TODO: it would be nice to have a sort of can_be_applied method
    '''
    node_to_expand = set()
    if hasattr(dace.libraries, 'linalg'):
        node_to_expand.add(dace.libraries.linalg.Inv)
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, tuple(node_to_expand)):
            node.implementation = "canonical"

    # TODO: make this better
    for node in sdfg.nodes()[0].nodes():
        if use_daceml and isinstance(node, donnx.nodes.onnx_op.ONNXMaxPool):
            node.implementation = "canonical"
        elif isinstance(node, dace.libraries.blas.Gemv):
            node.implementation = "canonical"


def enumerate_canonical_sdfg(sdfg: dace.SDFG,
                             state_id=0,
                             expansions: list = [],
                             flags: Set[str] = {'any'}) -> dace.SDFG:
    """
    Generator of Canonical SDFGs: given an SDFG returns all the canonical SDFGs (to be inlined for the sake of analyzability)
        that we can extract out of it, performing Application Space Exploration.

    :param sdfg: the application SDFG
    :param state_id: the state ID, defaults to 0
    :param expansions: list of expansions applied so far (node name, expansion name)
    :param flags: flags that must be supported by considered expansions
    :yield: a canonical SDFG
    """

    # first of all, expand the libnode until all of them are canonical expansion

    # These are the nodes that are not canonical byt heir own, and must be lowered into canonical ones
    non_canonical_nodes = {
        mmm.MMM,
        dace.libraries.blas.nodes.gemm.Gemm,
    }
    if use_daceml:
        non_canonical_nodes |= {onnx_op.ONNXConv, onnx_op.ONNXGemm, onnx_op.ONNXSoftmax, onnx_op.ONNXMatMul}
    if hasattr(dace.libraries, 'linalg'):
        non_canonical_nodes.add(dace.libraries.linalg.Inv)

    state = sdfg.states()[state_id]
    implementations = dict()
    for node, _ in state.all_nodes_recursive():
        # apply only to the first libnode that you find
        # TODO: this may be expensive, needs to be re-designed.
        if isinstance(node, tuple(non_canonical_nodes)):
            if use_daceml and issubclass(node.__class__, onnx_op.ONNXOp):
                # DACEML node, use canonical one or pure
                if "canonical" in node.implementations:
                    implementations["canonical"] = node.implementations["canonical"]
                else:
                    implementations["pure"] = node.implementations["pure"]

            else:
                if isinstance(node, mmm.MMM):
                    # TODO: fix this
                    implementations = node.implementations
                else:
                    # select all implementations that are canonical
                    for impl in node.implementations:
                        if impl.startswith("canonical"):
                            implementations[impl] = node.implementations[impl]

            for impl in implementations:

                sdfg_copy = copy.deepcopy(sdfg)
                state_copy = sdfg_copy.states()[state_id]
                node_id = state.node_id(node)
                node_copy = state_copy.nodes()[node_id]
                node_copy.implementation = impl

                if 'any' not in flags:
                    # if the expansion has the can_be_used method, check if this can be applied
                    can_be_used = getattr(implementations[impl], "can_be_used", None)
                    if callable(can_be_used):
                        if not can_be_used(flags):
                            # go to the next one
                            continue
                node_copy.expand(sdfg_copy, state_copy)

                #inlined = sdfg_copy.apply_transformations_repeated([InlineSDFG], print_report=True)
                # Use passes to Inline SDFG (less expensive)
                inlined = InlineSDFGs().apply_pass(sdfg_copy, {})

                assert inlined > 0, "Failed to expand and fully inline a non-canonical library node"

                expansions.append((node_copy.name, impl))
                yield from enumerate_canonical_sdfg(sdfg_copy, expansions=expansions, flags=flags)
                expansions.pop()
            break

    if 'sdfg_copy' in locals():
        return  # otherwise return the last one multiple times
        yield sdfg_copy, expansions
    else:
        yield sdfg, expansions


def compute_in_volume_special_node(edge,
                                   node,
                                   state,
                                   sdfg,
                                   fully_expanded_state=None,
                                   fully_expanded_sdfg=None,
                                   pred=None):
    """
    While building the Canonical DAG, we need to understand the amount of data being read by the nodes (tasks).
    There are cases where it is not possible to do so by just looking at the SDFG and the data movement analysis.

    This function deals with all the cases where this is the case, allowing to define "data analysis" for certain node types. 

    :param edge: input edge to the node
    :param node: node to be analyzed
    :param state: the state
    :param sdfg: the Canonical SDFG
    :param fully_expanded_state: fully expanded and inlined state
    :param fully_expanded_sdfg: fully expanded and inlined SDFG
    :param pred: predecessor
    :return: the input volume
    """

    if isinstance(node, reduction.ReduceMMM):
        # In this case the input volume is equal to the output one
        # (note that in this case the reduce node reads NMK elements, while
        # it should be NM)

        out_edge = state.out_edges(node)[0]
        return int(out_edge.data.volume)
    elif use_daceml and isinstance(node, donnx.nodes.onnx_op.ONNXReshape):
        if edge.dst_conn == 'shape':  # this edge will be removed from dace
            return 0
        else:
            acc_node = find_access_node_by_name(pred.label, fully_expanded_sdfg)
            exp_edge = fully_expanded_state.out_edges(acc_node)[
                0]  # TODO Here, we need to find the edge to which this node is connected, not a random one.
            volume = int(exp_edge.data.volume)
            return volume
    elif use_daceml and isinstance(node, donnx.nodes.onnx_op.ONNXSlice):
        # For Slicing we have two options: either we let them receive all the data and filter it
        # or the input must be in a buffer node.
        # Currently we are following the first approach, and for this we need to look at the non-expanded sdfg
        # (in the expanded one, the slice could be a copy between access nodes)
        if edge.dst_conn != 'data':
            volume = 0
        else:
            volume = int(edge.data.volume)
        return volume

    elif isinstance(node, linalg.Cholesky):

        assert node.implementation == "seq"
        # get input shape of A
        in_data = state.in_edges(node)[0].src
        N = int(in_data.desc(state).shape[0])

        # given the particular implementation, this node will be accessed 1/2(N+1)(N+2) times
        volume = (N + 1) * (N + 2) // 2
        return volume
    elif isinstance(node, forward_substitution.ForwardSubstitution):
        assert node.implementation == "seq"
        # get input shape of A
        in_data = state.in_edges(node)[0].src
        N = int(in_data.desc(state).shape[0])
        # given that implementation, the output is read around 1/6(N)(N+1)(N+2) times
        volume = (N + 1) * (N + 2) * N // 6
        return volume
    else:
        raise RuntimeError("Node type not found ", node)


def compute_out_volume_special_node(edge,
                                    node,
                                    state,
                                    sdfg,
                                    fully_expanded_state=None,
                                    fully_expanded_sdfg=None,
                                    pred=None):
    """
    While building the Canonical DAG, we need to understand the amount of data being written by the nodes (tasks).
    There are cases where it is not possible to do so by just looking at the SDFG and the data movement analysis.

    This function deals with all the cases where this is the case, allowing to define "data analysis" for certain node types. 

    :param edge: output edge to the node
    :param node: node to be analyzed
    :param state: the state
    :type state: the Canonical state (with libnodes that can be mapped 1:1 to canonical tasks)
    :param sdfg: the Canonical SDFG
    :param fully_expanded_state: fully expanded and inlined state
    :param fully_expanded_sdfg: fully expanded and inlined SDFG
    :param pred: predecessor
    :return: the output volume
    """

    if isinstance(node, linalg.Cholesky):

        # Note: if we don't consider the fact that this is also reading from the output result
        # this is modeled as an upsampler
        assert node.implementation == "seq"

        out_data = state.out_edges(node)[0].dst
        N = int(out_data.desc(state).shape[0])
        # given that implementation, the output is written 1/6(N+1)(N+2)(N+3) times
        volume = (N + 1) * (N + 2) * (N + 3) // 6

        return volume
    elif isinstance(node, forward_substitution.ForwardSubstitution):

        assert node.implementation == "seq"
        out_data = state.out_edges(node)[0].dst
        N = int(out_data.desc(state).shape[0])
        # given that implementation, the output is written around 1/6(N)(N+1)(N+2) times
        volume = (N + 1) * (N + 2) * N // 6
        return volume
    else:
        raise RuntimeError("Node type not found ", node)


def evaluate_canonical_sdfg(sdfg,
                            state_id,
                            num_pes,
                            expansions,
                            unroll_factor,
                            schedule_non_streaming=True,
                            on_chip_memory_sizes=[0],
                            print_schedule=False):
    """
    Given a Canonical SDFG evaluates it in terms of performance, power and area.

    For doing this:
    - the SDFG must be fully inlined
    - the corresponding Canonical DAG is created. While doing so, special attention is given to 
        situations where buffer node must be placed (detected using data movement analysis)

    :param sdfg: a canonical SDFG
    :param state_id: state id (0 by default)
    :param num_pes: list with the number of pes to consider
    :param expansions: list of node expansions applied to produce the SDFG
    :param unroll_factor: unrolling factor for iterative application
    :param schedule_non_streaming: whether or not schedule it also w/o streaming, defaults to True
    :param on_chip_memory_sizes: list of allowed on chip memory sizes that will be used for DSE
    :param print_schedule: whether or not printing the schedule on the standard output
    :return: a dictionary of results: num_pes -> result
    """

    # Special LibNodes for which the input or output volume is computed differently (e.g., reduction)
    #TODO: find uniform solution

    special_in_volume_nodes = {reduction.ReduceMMM, linalg.Cholesky, forward_substitution.ForwardSubstitution}
    if use_daceml:
        special_in_volume_nodes |= {onnx_op.ONNXSlice, onnx_op.ONNXReshape}

    special_out_volume_nodes = {linalg.Cholesky, forward_substitution.ForwardSubstitution}

    # Expand and inline the SDFG
    state = sdfg.states()[state_id]

    ### Get the top-level scope (if any) in the non-expanded SDFG
    scope_children = state.scope_children()
    scope_dict = state.scope_dict()
    # We already checked that there is only of them
    top_scopes = [n for n in scope_children[None] if isinstance(n, dace.sdfg.nodes.EntryNode)]

    ##### Expand and Inline SDFG
    # Create a copy to have also the non-expanded version
    fully_expanded_sdfg = copy.deepcopy(sdfg)
    start_time = time.time()
    fully_expanded_sdfg.expand_library_nodes()
    expansion_time = time.time() - start_time

    start_time = time.time()

    inlined = InlineSDFGs().apply_pass(fully_expanded_sdfg, {})

    inlining_time = time.time() - start_time
    fully_expanded_state = fully_expanded_sdfg.states()[state_id]

    ###### Build the canonical DAG

    resulting_dag = nx.DiGraph()
    print_debug_info = False  ## Prints debug info

    # Traverse the graph in topological order
    # Start from 1 to leave space for pseudo-root
    id = 1
    node_dictionary = {}  # dictionary libnode -> node id
    pseudo_nodes = {0}

    input_volumes_to_source = dict()
    access_node_to_buffer = {
    }  # dict access node name to buffer node (if this access node must be mapped to a buffer node)

    node_types = set()  # keep track of the different type of node found

    iterative_scope = False  # keep track if we are currently in an iterative scope (a map)
    previous_iterative_scope = None  # keep track of the previous iterative scope (may be necessary to track things down)

    for node in dace.sdfg.utils.dfs_topological_sort(state, sources=state.source_nodes()):
        if isinstance(node, dace.sdfg.nodes.LibraryNode):

            # Add the node:
            resulting_dag.add_node(id, name=id, label=f"{node.name}({id})")
            node_id = id
            node_dictionary[node] = node_id
            id += 1

            # Look at the predecessors:
            # 1. at first we will find access node: if this is planned to be a buffer node, connect the node to the buffer node
            # 2. otherwise, look at the parent of the predecessor, it should be a libnode, connect this there
            # Temp safety check: regarding point 2, the access node must have only one predecessor and it must be
            #   a library node

            for edge in state.in_edges(node):
                pred = edge.src
                through_map = False
                if not iterative_scope:
                    assert isinstance(pred, dace.sdfg.nodes.AccessNode)
                else:
                    mpath = state.memlet_path(edge)

                    # We should be able to look at the source (mpath[0]) and most closer edge
                    # But I am not sure how this would work for graph that are expanded more.

                    pred = mpath[0].src
                    through_map = isinstance(
                        mpath[-1].src, dace.sdfg.nodes.EntryNode
                    )  # Always? #TODO: check that we are going through a map, along the entire mpath?

                if pred.label in access_node_to_buffer:
                    # buffer node, add and edge with the right volume
                    # by looking at the same edge in the fully expanded sdfg

                    if isinstance(node, tuple(special_in_volume_nodes)):
                        volume = compute_in_volume_special_node(edge, node, state, sdfg, fully_expanded_state,
                                                                fully_expanded_sdfg, pred)
                    else:
                        # start from the access node in the fully expanded sdfg (we know the expansion
                        # will preserve it)
                        acc_node = find_access_node_by_name(pred.label, fully_expanded_sdfg)
                        if acc_node == None and print_debug_info:
                            print("Access node ", pred.label, " not found!!!")

                        if not through_map:
                            # Here, we need to find the edge to which this node is connected, not a random one.
                            # TODO: currently we look at the connectors, but this is not reliable, as there could be
                            # name clashes

                            for exp_edge in fully_expanded_state.out_edges(acc_node):
                                if exp_edge.dst_conn is not None and exp_edge.dst_conn.endswith(edge.dst_conn):
                                    break

                            else:
                                # it could be a view or an access-node to access node copy
                                if not isinstance(exp_edge.dst, dace.sdfg.nodes.AccessNode) and not isinstance(
                                        exp_edge.dst.desc(fully_expanded_state), dace.data.View):
                                    import pdb
                                    pdb.set_trace()
                                    raise RuntimeError("Don't know how to deal with this")

                            volume = int(exp_edge.data.volume)
                        else:
                            # We look at the memlet that goes through the iterative map, following the memlet path

                            for exp_edge in fully_expanded_state.out_edges(acc_node):
                                # We use the memlet tree, since there could be only one edge from
                                # access node and iterative map, but, in the scope, there could be more than one

                                # Note: the first edge in the memlet tree should be the one
                                # that lands in the map entry, the children the one that arrives to the library node(now expanded)
                                mtree = fully_expanded_state.memlet_tree(exp_edge)
                                assert isinstance(mtree.edge.dst, dace.sdfg.nodes.EntryNode)

                                found = False
                                # TODO: make this iterative: we don't know how deep we should go in the tree
                                for child in mtree.children:
                                    if child.edge.dst_conn is not None and child.edge.dst_conn.endswith(edge.dst_conn):
                                        found = True
                                        break
                                if found:
                                    break
                            else:
                                # TODO: use memlet_tree in case this is an access node that goes into the map
                                import pdb
                                pdb.set_trace()
                                raise RuntimeError("Can not follow mpath")

                            volume = int(child.edge.data.volume)
                            # TODO: consolidate it, can be fragile, there is no way to guarantee this is correct

                    if volume != 0:
                        resulting_dag.add_edge(access_node_to_buffer[pred.label], node_id, weight=volume)
                    if print_debug_info:
                        print(f"Connecting {node.name} to previous buffer node {pred.label}")
                else:
                    # connect this to the previous library node(s)
                    # TODO: deal with Access Node coming from expansions
                    assert pred.desc(state).transient or state.in_degree(pred) <= 1

                    if iterative_scope:
                        mpath = state.memlet_path(edge)

                    for pred_ln in state.predecessors(pred):

                        # The predecessor (of the predecessor) must be a library node, or an access node (in case of access node to access node copies)
                        if not isinstance(pred_ln, (dace.sdfg.nodes.LibraryNode, dace.sdfg.nodes.AccessNode)):
                            #TODO: we could be inside an iterative scope?
                            import pdb
                            pdb.set_trace()
                        assert isinstance(
                            pred_ln,
                            (dace.sdfg.nodes.LibraryNode, dace.sdfg.nodes.AccessNode
                             )), f"{pred_ln}, the predecessor of {node}, is not a library node nor an access node"

                        # if the predecessor node is a "special" one, compute its output volume with ad-hoc function
                        if isinstance(node, tuple(special_in_volume_nodes)):
                            volume = compute_in_volume_special_node(edge, node, state, sdfg, fully_expanded_state,
                                                                    fully_expanded_sdfg, pred)
                        else:
                            #TODO: deal with multiple in/out edges properly (for doing so we need to understand what is the corresponding edge
                            # in the fully expanded SDFG). This now can be wrong

                            # TODO: what are we looking for here?
                            acc_node = find_access_node_by_name(pred.label, fully_expanded_sdfg)
                            if acc_node.desc(state).transient:
                                assert not edge.data.dynamic, f"Edge {edge} has a dynamic volume"
                                volume = int(edge.data.volume)
                            else:

                                # Here, we need to find the edge to which this node is connected, not a random one.
                                # TODO: currently we look at the connectors, but this is not reliable, as there could be
                                # name clashes

                                for exp_edge in fully_expanded_state.out_edges(acc_node):
                                    if exp_edge.dst_conn.endswith(edge.dst_conn):
                                        break
                                # TODO: this happens in lenet
                                # else:
                                #     import pdb
                                #     pdb.set_trace()

                                assert not exp_edge.data.dynamic, f"Edge {exp_edge} has a dynamic volume"
                                volume = int(exp_edge.data.volume)
                        if isinstance(pred_ln, dace.sdfg.nodes.LibraryNode):
                            pred_node = node_dictionary[pred_ln]
                            # resulting_dag.add_edge(node_dictionary[pred_ln], node_id, weight=volume)
                        else:
                            # This could be a transient array that therefore we didn't consider as buffer node.
                            # Therefore we continue traversing back until we find the previous library node
                            # TODO: this is done just two times, generalize
                            if pred_ln.label not in access_node_to_buffer:
                                assert state.in_degree(pred_ln) == 1
                                pred_pred = state.predecessors(pred_ln)[0]
                                if not isinstance(pred_pred, dace.sdfg.nodes.LibraryNode):
                                    import pdb
                                    pdb.set_trace()

                                assert isinstance(pred_pred, dace.sdfg.nodes.LibraryNode)
                                pred_node = node_dictionary[pred_pred]
                            else:
                                assert pred_ln.label in access_node_to_buffer
                                pred_node = access_node_to_buffer[pred_ln.label]
                        if volume != 0:
                            resulting_dag.add_edge(pred_node, node_id, weight=volume)

                        if print_debug_info:
                            print(
                                f"Connecting {node.name} to previous Library Node {pred_ln.name}, with volume {volume}")

            continue

        elif isinstance(node, dace.sdfg.nodes.EntryNode):
            # check that this is a top-level scope
            if node in top_scopes:
                if print_debug_info:
                    print(node, " is a top_level scope")
                iterative_scope = True
                assert previous_iterative_scope == None  # only one top-level map
            else:
                RuntimeError("NYI: found a map scope but this is not a top-level one")

        elif isinstance(node, dace.sdfg.nodes.ExitNode):
            if scope_dict[node] in top_scopes:
                if print_debug_info:
                    print(node, " is the exit top_level scope")
                iterative_scope = False
                previous_iterative_scope = node
            else:
                RuntimeError("NYI: found a map scope but this is not a top-level one")

        elif isinstance(node, dace.sdfg.nodes.AccessNode):
            buffer_node = is_buffer_node(node, state, fully_expanded_sdfg)
            through_map = False
            if buffer_node:
                # does this exist in the final sdfg? if no, no need to add it
                acc_node = find_access_node_by_name(node.label, fully_expanded_sdfg)
                if acc_node != None:
                    if print_debug_info:
                        print(f"Acc Node {node.label} is a buffer node")

                    # add it and connect to the previous ln
                    resulting_dag.add_node(id, name=id, label=f"{node.label}_buffer({id})")
                    access_node_to_buffer[node.label] = id

                    # get predecessor
                    assert state.in_degree(node) <= 1, f"Access node {node.label} has more than one input"
                    if state.in_degree(node) > 0:
                        in_edge = state.in_edges(node)[0]
                        predecessor = in_edge.src

                        if (isinstance(predecessor, dace.nodes.EntryNode) or isinstance(
                                predecessor, dace.nodes.ExitNode)) and (iterative_scope
                                                                        or previous_iterative_scope is not None):
                            mpath = state.memlet_path(in_edge)

                            # We are arriving to this access node through an iterative scope: either this access node is just
                            # after the exit scope (previous_iterative_scope is not None), or this is inside and connected to an access node outside
                            if (previous_iterative_scope is None and
                                    isinstance(mpath[-1].src,
                                               dace.sdfg.nodes.EntryNode)) or mpath[-1].src == previous_iterative_scope:
                                through_map = True
                                predecessor = mpath[-2].src  # the edge that arrives to the map exit
                            else:
                                import pdb
                                pdb.set_trace()
                                assert False
                        # get the right input volume by looking at the expanded sdfg
                        # acc_node = find_access_node_by_name(node.label, fully_expanded_sdfg)
                        if fully_expanded_state.in_degree(acc_node) == 0:
                            import pdb
                            pdb.set_trace()
                        exp_edge = fully_expanded_state.in_edges(acc_node)[0]

                        if isinstance(predecessor, tuple(special_out_volume_nodes)):
                            volume = compute_out_volume_special_node(edge, predecessor, state, sdfg,
                                                                     fully_expanded_state, fully_expanded_sdfg)
                        else:
                            assert not exp_edge.data.dynamic, f"Edge {exp_edge} has a dynamic volume"
                            if not through_map:
                                volume = int(exp_edge.data.volume)
                            else:
                                # get the actual volume, from inside the iterative scope
                                mpath_exp = fully_expanded_state.memlet_path(exp_edge)

                                assert isinstance(mpath_exp[-1].src, dace.sdfg.nodes.ExitNode) or isinstance(
                                    mpath_exp[-1].src, dace.sdfg.nodes.EntryNode)
                                if isinstance(mpath[-1].src, dace.sdfg.nodes.ExitNode):
                                    # this access node is just after the exit node, look at the memlet inside the scope
                                    volume = int(mpath_exp[-2].data.volume)
                                else:
                                    # this access node is just after the entry node, look at the memlet inside the scope
                                    volume = int(mpath_exp[-1].data.volume)

                        if isinstance(predecessor, dace.sdfg.nodes.LibraryNode):
                            pred_node = node_dictionary[predecessor]
                        elif isinstance(predecessor, dace.sdfg.nodes.AccessNode):
                            # access node  to access node
                            if predecessor.label in access_node_to_buffer:
                                # buffer node
                                pred_node = access_node_to_buffer[predecessor.label]
                            else:
                                # connect to the previous library node (or access node in case of access-node to access- node copy)
                                #let's assume for now that there is only one
                                assert state.in_degree(predecessor) == 1
                                pred_ln = state.predecessors(predecessor)[0]

                                assert isinstance(pred_ln, (dace.sdfg.nodes.LibraryNode, dace.sdfg.nodes.AccessNode))
                                assert predecessor.desc(state).transient

                                # This could be a transient array that therefore we didn't consider as buffer node.
                                # Therefore we continue traversing back until we find the previous library node
                                # TODO: this is done just two times, generalize

                                if isinstance(pred_ln, dace.sdfg.nodes.LibraryNode):
                                    pred_node = node_dictionary[pred_ln]
                                else:
                                    if pred_ln.label not in access_node_to_buffer:
                                        assert state.in_degree(pred_ln) == 1
                                        pred_pred = state.predecessors(pred_ln)[0]
                                        if not isinstance(pred_pred, dace.sdfg.nodes.LibraryNode):
                                            import pdb
                                            pdb.set_trace()

                                        assert isinstance(pred_pred, dace.sdfg.nodes.LibraryNode)
                                        pred_node = node_dictionary[pred_pred]
                                    else:
                                        assert pred_ln.label in access_node_to_buffer
                                        pred_node = access_node_to_buffer[pred_ln.label]

                                # the volume is already computed

                        else:
                            import pdb
                            pdb.set_trace()
                        resulting_dag.add_edge(pred_node, id, weight=volume)

                        if print_debug_info:
                            print(f"Connecting access node {node.label} to {predecessor.label}, with volume {volume}")

                    id += 1

            else:
                if print_debug_info:
                    print(f"Acc Node {node.label} is NOT buffer node")
        else:

            raise RuntimeError(
                f"NYI {node}: for the moment being, the SDFG must contain only LibraryNodes and AccessNodes")

    ### At this point we've built  the dag. We need to finalize it

    # 1. if this dag is within a map-top level scope, unroll it
    # (currently totally, map scope must be a number)
    num_iterations = 1
    if len(top_scopes) > 0:
        num_iterations = top_scopes[0].range.size()[0]
        assert unroll_factor <= num_iterations
        assert num_iterations % unroll_factor == 0, "Unroll factor must evenly divide the number of iterations"
        unrolled_nodes = defaultdict(list)
        # continue from last id defined before
        if len(top_scopes) == 1:
            # TODO: currently if a top-level map exists, this includes everything
            # TODO: it would be nice to only replicate the meaningful part (e.g. no access nodes that are not within the top level scope)
            # TODO: this would require a lot of testing to be consistent

            not_in_top_scope = set(state.nodes()) - set(scope_children[top_scopes[0]]) - {top_scopes[0]}
            nodes_in_scope = set(scope_children[top_scopes[0]])

            # Idea:
            # - traverse in topo order
            # - if the node correspond to a node not in top_scope, go on
            # - otherwise, create a replica
            # - in the meanwhile build a dictionary: original node -> unrolled nodes
            # - go over the original edges, and for each node, create in/out edges for all the replicas

            # traverse the dag in topological-order
            for node in nx.algorithms.topological_sort(resulting_dag):
                if node in access_node_to_buffer.values():
                    continue
                # get the SDFG node and check if is not in top_scope
                node_label = resulting_dag.nodes(data=True)[node]['label']
                # create the replica
                for i in range(unroll_factor - 1):
                    resulting_dag.add_node(id, label=f"{node_label}_{i+1}")
                    unrolled_nodes[node].append(id)
                    # print(f"Duplicated node {node} ({node_label}), to: {id}")
                    id += 1
            for src, dst, data in list(resulting_dag.edges(data=True)):
                if src in unrolled_nodes and dst in unrolled_nodes:
                    srcs = unrolled_nodes[src]
                    dsts = unrolled_nodes[dst]
                    if len(srcs) != len(dsts):
                        import pdb
                        pdb.set_trace()

                    assert len(srcs) == len(dsts)
                    for i in range(len(srcs)):
                        resulting_dag.add_edge(srcs[i], dsts[i], weight=data['weight'])
                elif src in unrolled_nodes and dst not in unrolled_nodes:
                    srcs = unrolled_nodes[src]
                    for i in range(len(srcs)):
                        resulting_dag.add_edge(srcs[i], dst, weight=data['weight'])
                        # print("*Adding edge: ", srcs[i], dst)
                elif src not in unrolled_nodes and dst in unrolled_nodes:
                    dsts = unrolled_nodes[dst]
                    for i in range(len(dsts)):
                        resulting_dag.add_edge(src, dsts[i], weight=data['weight'])

    # There could be that some buffer node does not have any input/output edges. In case we can just prune it
    # TODO: this may give problem with the scheduling that is expecting nodes with increasing id
    pruned = []
    for k, v in access_node_to_buffer.items():
        if resulting_dag.in_degree(v) == 0 and resulting_dag.out_degree(v) == 0:
            # print("Removing isolated buffer node: ", k, v)
            resulting_dag.remove_node(v)
            pruned.append(k)
    for p in pruned:
        del access_node_to_buffer[p]

    # if there are more than 1 source node, add a pseudo-source, with zero cost
    source_node = [node for node in resulting_dag.nodes if resulting_dag.in_degree(node) == 0]
    # Note: all source nodes are buffer nodes here
    if len(source_node) > 1:
        resulting_dag.add_node(0, name=0, label=f"pseudo-source({id})")
        # node_dictionary["pseudo-source"]["id"] = 0
        for sn in source_node:
            # resulting_dag.add_edge(0, sn, weight=input_volumes_to_source[sn])
            resulting_dag.add_edge(0, sn, weight=list(resulting_dag.out_edges(sn, data=True))[0][2]['weight'])
    else:
        resulting_dag.add_edge(0, 1, weight=list(resulting_dag.out_edges(1, data=True))[0][2]['weight'])

    # do the same for exit node
    exit_node = [node for node in resulting_dag.nodes if resulting_dag.out_degree(node) == 0]
    pseudo_exit_node = -1
    if len(exit_node) > 1:
        resulting_dag.add_node(id, name=id, label=f"pseudo-exit({id})", pseudo=True, weight=0)
        pseudo_nodes.add(id)
        pseuo_exit_node = id
        id += 1
        for en in exit_node:
            resulting_dag.add_edge(en, id, weight=0)

    ########### Evaluate the canonical DAG
    results = defaultdict(list)  # dictionary num_pes -> corresponding result

    #### Schedule

    for np in num_pes:
        # visualize_dag(resulting_dag, node_labels_attribute="label")
        # visualize_dag(resulting_dag)

        start_time = time.time()
        scheduler = StreamingScheduler(resulting_dag, num_pes=np, buffer_nodes=access_node_to_buffer.values())
        streaming_paths_str_int, streaming_components_str_int = spatial_block_partitioning.spatial_block_partitioning(
            resulting_dag,
            np,
            0,
            pseudo_exit_node,
            buffer_nodes=access_node_to_buffer.values(),
            create_new_blocks=False)

        set_streams_from_streaming_paths(resulting_dag, streaming_paths_str_int)

        # visualize_dag(resulting_dag, node_labels_attribute="label")
        scheduler.streaming_interval_analysis()

        pes_schedule_gang_str_int, tasks_schedule_gang_str_int = scheduler.gang_schedule(streaming_components_str_int)
        # Non-gang schedule
        # pes_schedule_gang_str_int, tasks_schedule_gang_str_int = scheduler.schedule_dag(streaming_components_str_int)

        streaming_makespan = makespan(tasks_schedule_gang_str_int)

        ### compute buffer space
        # TODO: refine and verify the result

        from streamingsched.sched.deadlock_prevention import compute_buffer_space

        channels_capacities = compute_buffer_space(resulting_dag, streaming_components_str_int,
                                                   tasks_schedule_gang_str_int, source_node)

        # reiterate buffer space computation to refine it
        # removing steraming edges with weight > buffer_space (it is not streaming)

        continue_check = True
        iterations = 0
        while continue_check:
            remove_edges = []
            for src, dst, data in resulting_dag.edges(data=True):
                if 'stream' in data and data['stream']:
                    if data['weight'] > 1 and data['weight'] - 1 <= channels_capacities[src, dst]:
                        # print("Removing ", (src, dst))
                        resulting_dag.edges[src, dst]['stream'] = False
                        remove_edges.append((src, dst))
                    elif data['weight'] == 1:
                        # print("Removing reducer edge", (src, dst))
                        resulting_dag.edges[src, dst]['stream'] = False
                        remove_edges.append((src, dst))

            if len(remove_edges) > 0:

                scheduler = StreamingScheduler(resulting_dag, num_pes=np, buffer_nodes=access_node_to_buffer.values())
                scheduler.streaming_interval_analysis()
                # TODO: ideally we should recompute the streaming blocks here, but we need to this in the right way

                pes_schedule_gang_str_int, tasks_schedule_gang_str_int = scheduler.gang_schedule(
                    streaming_components_str_int)
                streaming_makespan = makespan(tasks_schedule_gang_str_int)

                channels_capacities = defaultdict(lambda: 1)
                channels_capacities = compute_buffer_space(resulting_dag, streaming_components_str_int,
                                                           tasks_schedule_gang_str_int, source_node)

            else:
                continue_check = False
            iterations += 1

        exc = [(e, v) for e, v in channels_capacities.items() if v > 32]

        # if len(exc) > 1:
        #     print("------------------ Exceeding edges: ", exc)

        #     print("********************************************************************")
        #     for e, v in exc:
        #         print(resulting_dag.nodes()[e[0]], resulting_dag.nodes()[e[1]])
        #     print("********************************************************************")

        # NOTE: if an edge is streaming we always assume the buffer space is 1.
        # Therefore we count only the case where the computed buffer space is greater than 1
        buffer_space_deadlock = 0
        for k, v in channels_capacities.items():
            if v > 1:
                buffer_space_deadlock += v

        if len(top_scopes) > 0 and num_iterations > 1:
            # We need to update the makespan, since the application execution will be repeated multiple times.
            # We need to take into account also the unrolling_factor
            streaming_makespan = streaming_makespan * num_iterations / unroll_factor

        scheduling_time = time.time() - start_time

        if print_schedule:
            print_sched(pes_schedule_gang_str_int, "PEs")

        ### Non Streaming Schedule
        non_streaming_makespan = 0
        if schedule_non_streaming:
            W = build_W_matrix_HEFT(resulting_dag, 0, pseudo_exit_node, access_node_to_buffer.values(), 1)
            heft_pes_schedule, heft_tasks_schedule = eft.schedule_dag(resulting_dag, W, np)
            non_streaming_makespan = int(makespan(heft_tasks_schedule))

            if len(top_scopes) > 0 and num_iterations > 1:
                # We need to update the makespan, since the application execution will be repeated multiple times.
                # We need to take into account also the unrolling_factor
                non_streaming_makespan = non_streaming_makespan * num_iterations / unroll_factor

        ##### On-Chip buffer space exploration for Buffer nodes
        # We test now various options for the on-chip memory area
        # For each of them, we map the buffer node of each spatial block in that area (TODO: opt problem)
        # and we count accordingly the number of off-chip and on-chip I/Os
        # TODO: have the memory are be organized in memories of given size

        buffer_nodes = set(access_node_to_buffer.values())
        for obs in on_chip_memory_sizes:
            # Map each buffer node on the on-chip memory area
            buffer_nodes_on_chip = set()  # set of buffer nodes that are mapped on the on-chip area

            for sb in streaming_components_str_int:

                # get the buffer nodes in this spatial block
                bn_in_block = list(buffer_nodes.intersection(sb))

                # Map the buffer nodes in the block to the available on-chip area
                # This is another optimization problem, similar to knapsack: we want
                # to map the buffer nodes that will give us the max off-chip I/Os reduction
                # TODO: consider whole on-chip area to be partitioned across multiple smaller memories
                # (this is an example of 'Multiple Knapsack problem')

                weights = [get_buffer_node_space(resulting_dag, bn) for bn in bn_in_block]
                values = [get_data_movements_node(resulting_dag, bn) for bn in bn_in_block]

                ids = knapsack(obs, weights, values)

                for id in ids:
                    buffer_nodes_on_chip.add(bn_in_block[id])

                # get the buffer nodes that are included

            # Count on-chip and off-chip I/Os, taking into account mapping

            off_chip_ios, on_chip_ios, streaming_ios, buffer_space, buffer_nodes_space_histogram = count_io_canonical_dag_global(
                resulting_dag, access_node_to_buffer.values(), pseudo_nodes, buffer_nodes_on_chip)

            # other statistics
            buff_space_per_block, ios_per_block, ios_btw_blocks = count_io_canonical_dag_local(
                resulting_dag, access_node_to_buffer.values(), pseudo_nodes, streaming_components_str_int)

            dynamic_power, static_power, performance, area = ppa_model_base(streaming_makespan, np, off_chip_ios,
                                                                            on_chip_ios, obs)

            # Save the result
            r = Result(expansion_list=expansions.copy(),
                       num_pes=np,
                       on_chip_space=obs,
                       makespan=streaming_makespan,
                       ns_makespan=non_streaming_makespan,
                       static_power=static_power,
                       dynamic_power=dynamic_power,
                       performance=performance,
                       area=area,
                       unroll_factor=unroll_factor,
                       num_iterations=num_iterations,
                       off_chip_IOs=off_chip_ios,
                       on_chip_IOs=on_chip_ios,
                       streaming_IOs=streaming_ios,
                       time_expansion=expansion_time,
                       time_inlining=inlining_time,
                       time_scheduling=scheduling_time,
                       number_of_nodes=resulting_dag.number_of_nodes(),
                       number_of_buffer_nodes=len(access_node_to_buffer.values()),
                       buffer_space_deadlock=buffer_space_deadlock,
                       buffer_space=buffer_space,
                       buffer_nodes_space_histogram=buffer_nodes_space_histogram,
                       number_of_blocks=len(streaming_components_str_int),
                       buffer_space_per_block=buff_space_per_block,
                       ios_per_block=ios_per_block,
                       ios_btw_blocks=ios_btw_blocks)

            results[np].append(r)
    return results


def worker(queue,
           worker_id,
           state_id,
           num_pes,
           results,
           unroll_factor=1,
           on_chip_memory_sizes=[0],
           print_schedule=False):
    ''' 
        Multithreaded implementation, works on a single canonical SDFG
    '''
    while (True):
        i, sdfg, expansions = queue.get()
        if i == -1:
            break

        r = evaluate_canonical_sdfg(sdfg,
                                    state_id,
                                    num_pes,
                                    expansions,
                                    unroll_factor,
                                    on_chip_memory_sizes=on_chip_memory_sizes,
                                    print_schedule=print_schedule)
        for np in num_pes:
            results[np].extend(r[np])
        # print(f"[{worker_id}]: Expansion: {i}: {expansions}")

    print("Worker ", worker_id, " finished!")


def DSE(orig_sdfg: dace.SDFG,
        num_pes: list = [8],
        on_chip_memory_sizes=[0],
        use_multithreading=False,
        n_threads=8,
        unroll_factor=1,
        supported_flags: Set[str] = {'any'},
        state_id=0,
        store_csv=True,
        print_schedule=False) -> dict:
    """
    Given the application SDFG perform Design Space Exploration. This comprises:

    - Application Space Exploration: the SDFG is "compiled" to canonical DAG. Certain operations may have 
      different expansions, that would result in different streaming and parallelism opportunities
    
    -Architecture space exploration: considers architectures with different number of processing elements (PEs) and on-chip
        memory space, as hinted by the user
    
    NOTE: the DSE phase can take long time, during which no output is provided.
    TODO: implement progress bar.

    :param orig_sdfg: the application SDFG. This must be comprised only by a single state containing library nodes, access nodes, 
        and (at most) one top-scope (maps)
    :param num_pes: a list of integers that represent the number of PEs against which we schedule the resuling canonical DAG, defaults to [8]
    :param on_chip_memory_sizes: the allowed amount of on-chip area that will be used by buffer nodes. Multiple configurations can be indicated here
    :param use_multithreading: whether to use or not multithreading to perform DSE, defaults to False
    :param n_threads: number of threads used for DSE, defaults to 8
    :param unroll_factor: if the application is an iterative computation (contains a single top-level map-scope), we can partially unroll it to 
        increase the chance of exploiting parallelism while keeping computation time reasonable the defaults to 1
    :type unroll_factor: int, optional
    :param state_id: the state id , defaults to 0
    :param supported_flags: whenever applicable, uses only lib node expansions that support at least one of the flags here indicated. If 'any' is present
        (default) it means that all expansions will be considered.
    :param store_csv: whether or not store the result in CSV files, one for each considered number of PEs (True by default)
    :param print_schedule: whether or not print the schedule on the standard output (False by default to prevent flooding the standard output)
    :return:  a dictionary of num_pes -> [list of results]

    """

    if not isinstance(num_pes, list):
        num_pes = [num_pes]

    specialize_matmul(orig_sdfg)
    register_canonical_implementations()
    #Register implementations
    set_node_expansions(orig_sdfg)

    ####  Checks:

    # 1 - Currently we support DAGs that contain at more 1 map as top-level scope.
    #       We check this by looking at scope_children
    state = orig_sdfg.states()[state_id]
    scope_children = state.scope_children()
    scope_dict = state.scope_dict()
    top_scopes = [n for n in scope_children[None] if isinstance(n, dace.sdfg.nodes.EntryNode)]
    assert len(top_scopes) <= 1

    # 2 - ensure that map iterations are actually independent (no WCR)
    if len(top_scopes) == 1:
        exit_scope = [
            n for n in scope_children[top_scopes[0]]
            if isinstance(n, dace.sdfg.nodes.ExitNode) and scope_dict[n] == top_scopes[0]
        ]

        for out_edge in state.out_edges(exit_scope[0]):
            assert out_edge.data.wcr == None, "Can not have WCR"

    # 3 - if there is a top-level scope, than it should contain everything
    # TODO: relax this requirement
    if len(top_scopes) == 1:
        not_in_top_scope = set(state.nodes()) - set(scope_children[top_scopes[0]]) - {top_scopes[0]}
        assert all(isinstance(x, dace.sdfg.nodes.AccessNode) for x in not_in_top_scope)
    # 4 - TODO check that there is no nested map? We can do it by iterating over the children of the top scope

    ### Multithreading

    if use_multithreading:
        # Multiprocessing: each workers deal with a different DAG
        workers_queues = [Queue()] * n_threads

        # Now start all processes
        processes = []
        manager = Manager()
        results = manager.dict()
        for np in num_pes:
            results[np] = manager.list()
        for i in range(n_threads):
            processes.append(
                Process(target=worker,
                        args=(workers_queues[i], i, state_id, num_pes, results, unroll_factor, on_chip_memory_sizes,
                              print_schedule)))
            processes[i].start()
    else:

        # TODO: use named tuples
        results = dict()
        for np in num_pes:
            results[np] = []

    start_time = time.time()

    i = 0
    for sdfg, expansions in enumerate_canonical_sdfg(orig_sdfg, flags=supported_flags):

        if use_multithreading:
            # Multithreaded
            workers_queues[i % n_threads].put((i, sdfg, expansions.copy()))

        else:
            r = evaluate_canonical_sdfg(sdfg,
                                        state_id,
                                        num_pes,
                                        expansions,
                                        unroll_factor,
                                        on_chip_memory_sizes=on_chip_memory_sizes,
                                        print_schedule=print_schedule)
            for np in num_pes:
                results[np].extend(r[np])

        i += 1

    if use_multithreading:
        # Stop workers
        for i in range(n_threads):
            workers_queues[i].put((-1, -1, -1))

        for i in range(n_threads):
            workers_queues[i].close()
            workers_queues[i].join_thread()

        for i in range(n_threads):
            processes[i].join()

    overall_processing_time = time.time() - start_time
    td = timedelta(seconds=overall_processing_time)

    # The computed results are in a dictionary of lists (one per considered number of pes)

    for np in num_pes:
        np_results = list(results[np])
        np_results.sort(key=lambda x: x.makespan)

        results[np] = np_results  # store it back sorted

        if store_csv:
            ## Save results to file
            data = []
            for r in np_results:
                data.append([
                    r.makespan, r.off_chip_IOs, r.on_chip_IOs, r.streaming_IOs, r.dynamic_power, r.static_power,
                    r.performance, r.area, r.number_of_nodes, r.number_of_buffer_nodes, r.time_expansion,
                    r.time_inlining, r.time_scheduling, r.ns_makespan, r.buffer_space_deadlock, r.buffer_space,
                    list(r.buffer_nodes_space_histogram[0]),
                    [round(item) for item in r.buffer_nodes_space_histogram[1]
                     ], r.number_of_blocks, r.buffer_space_per_block, r.ios_per_block,
                    list(r.ios_btw_blocks[0]), [round(item) for item in r.ios_btw_blocks[1]], r.num_pes,
                    r.on_chip_space, r.expansion_list
                ])

            if np_results[0].num_iterations == 1 and np_results[0].unroll_factor == 1:
                results_filename = f'results_{sdfg.name}_npes_{np}.csv'
            else:
                results_filename = f'results_{sdfg.name}_npes_{np}_nit_{np_results[0].num_iterations}_unrolling_{np_results[0].unroll_factor}.csv'

            results_header = [
                "Makespan", "Off-Chip IOs", "On-Chip IOs", "Streaming IOs", "Dynamic Power Score", "Static Power Score",
                "Performance score", "Area score", "Total number of canonical nodes", "Number of buffer nodes",
                "SDFG expansion time", "SDFG inlining time", "Scheduling time", "Non-Streaming Makespan",
                "Buffer space for deadlock prev.", "Total Buffer space buff nodes", "Buffer Nodes space distribution",
                "Buffer Nodes Space Bins", "Number of blocks", "Buff Space per block", "IOs per block",
                "IOs btw blocks space distribution", "IOs btw blocks Space Bins", "Number of PEs", "On-Chip Memory"
                "Expansion list"
            ]

            save_results_to_file(results_filename, results_header, data)

    print('Processing time in hh:mm:ss:', td)

    return results
