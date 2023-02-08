# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
Collection of utility functions used for DSE
"""

import csv
import ast
from collections import defaultdict
from dse.data import Result
import dace
import networkx as nx
import numpy as np
import statistics


def save_results_to_file(filename: str, header: list, data):
    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


def load_results_from_file(dir: str):
    """
    Reads the results from csv files contained in the given directory

    """
    # we want to rebuild the original structure, which is a dict #pes -> list of results
    # TODO find a better structure?

    import glob, os
    os.chdir(dir)
    results = defaultdict(list)

    for filename in glob.glob("results*.csv"):

        with open(filename, encoding='UTF8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            # expected header
            results_header = [
                "Makespan", "IOs canonical DAG", "IOs SDFG", "Total number of canonical nodes",
                "Number of buffer nodes", "SDFG expansion time", "SDFG inlining time", "Scheduling time",
                "Non-Streaming Makespan", "Buffer space", "Number of PEs", "Expansion list"
            ]

            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    assert row == results_header
                    line_count += 1
                else:
                    # build the original result
                    np = int(row[10])

                    r = Result(
                        makespan=int(row[0]),
                        ns_makespan=int(row[8]),
                        num_pes=np,
                        unroll_factor=0,  # unknown, need to be saved
                        num_iterations=0,  # unknown need to be saved
                        IOs_sdfg=int(row[2]),
                        IOs_canonical=int(row[1]),
                        time_expansion=float(row[5]),
                        time_inlining=float(row[6]),
                        time_scheduling=float(row[7]),
                        number_of_nodes=int(row[3]),
                        number_of_buffer_nodes=int(row[4]),
                        buffer_space=int(row[9]),
                        expansion_list=ast.literal_eval(row[11]))
                    results[np].append(r)

                    line_count += 1

    for np in results.keys():
        np_results = list(results[np])
        np_results.sort(key=lambda x: x.makespan)

        results[np] = np_results  # store it back sorted

    return results


def set_streams_from_streaming_paths(dag, streaming_paths):
    '''
        Given a DAG set edges to streaming
    '''
    for t in streaming_paths:
        for i in range(len(t) - 1):
            if dag.has_edge(t[i], t[i + 1]):
                # print("Setting: ", t[i], t[i + 1])
                dag[t[i]][t[i + 1]]['stream'] = True
            else:
                print(f"Something is going wrong: edge ({t[i]}, {t[i+1]}) does not exists.")


def get_access_node_predecessors(node, graph):

    acc_predecessors = []
    for edge in graph.in_edges(node):
        # check that this edge has a predecessor
        pred = edge.src
        if isinstance(pred, dace.sdfg.nodes.AccessNode):
            acc_predecessors.append(pred)


def find_access_node_by_name(name: str, sdfg: dace.SDFG):
    """
    Returns the access node by looking up for its name

    """
    for candidate, _ in sdfg.all_nodes_recursive():
        if isinstance(candidate, dace.nodes.AccessNode):
            if candidate.label == name:
                return candidate
    # import pdb
    # pdb.set_trace()
    # raise RuntimeError(f"Not found access node with name {name}")
    return None


def count_io(fully_expanded_sdfg: dace.SDFG, fully_expanded_state: dace.SDFGState, buffer_node_labels: list):
    """
    Given a fully expanded SDFG, counts the I/Os from buffer nodes (which represent external memory accesses)

    :param fully_expanded_sdfg: the SDFG
    :param fully_expanded_state: the state
    :param buffer_node_labels: list of buffer node names
    :return: number of I/Os counted as volume (number of elements of the user-defined type)
    """

    reads = 0
    writes = 0
    for bn in buffer_node_labels:
        # find the access node
        anode = find_access_node_by_name(bn, fully_expanded_sdfg)

        # TODO: what to count? bytes?
        node_reads = 0
        node_writes = 0

        # get all input and output edges
        for edge in fully_expanded_state.in_edges(anode):
            memlet = edge.data
            node_writes += memlet.volume
        for edge in fully_expanded_state.out_edges(anode):
            memlet = edge.data
            node_reads += memlet.volume

        # print(f"*** Node {bn}: total I/O = {node_reads+node_writes} ({node_reads} reads, {node_writes} writes)")
        reads += node_reads
        writes += node_writes

    return reads + writes


def get_buffer_node_space(dag: nx.DiGraph, bn: int):
    '''
        Given a buffer node returns its buffer space by looking at 
        its input edges
    '''

    buff_data = -1

    for src, dst, data in dag.in_edges(bn, data=True):

        if buff_data == -1:
            buff_data = data["weight"]
        else:
            if buff_data != data["weight"]:
                print("**********************Buffer node ", bn, " ", dag.nodes(bn),
                      "has more than one input edge with different volumes")
                raise RuntimeError("Not ok!")
    assert buff_data != -1  # sanity check
    return buff_data


def get_data_movements_node(dag: nx.DiGraph, node: int):
    """
    Given a node, counts the total volume of data movements that
    involves it (either movements to the node, or from the node)

    """

    volume = 0
    for _, _, data in dag.in_edges(node, data=True):
        volume += data['weight']

    for _, _, data in dag.out_edges(node, data=True):
        volume += data['weight']

    return volume


def count_io_canonical_dag_global(dag: nx.DiGraph, buffer_nodes: set, pseudo_nodes: set, buffer_nodes_on_chip: set):
    '''
        Given a Canonical DAG, with streaming/nonstreaming edges, and a mapping of buffer nodes into on-chip memory,
        it counts the number of off-chip and on-chip IOs by looking at the entire DAG.
        
        This is essentially the number of elements that are read and/or
        written in a non-streaming edge. 
        **NOTE** :This accounts also for the edges that are coming out
        from a buffer node, for which we also consider the on-chip I/Os

        **Note**: data movements of streaming edges are not counted as I/Os

        This procedure returns:
        - the total number of off-chip and on-chip I/Os, as described before
        - the total number of streaming_ios, i.e., data movements that occur on streaming edges
            between computational nodes
        - the total buffer space: this is the sum of amounts of data that enters a buffer node
        - an np.histogram for buffer node spaces distribution

    '''

    # And we do also some checks
    on_chip_io_reads = 0
    off_chip_io_reads = 0
    on_chip_io_writes = 0
    off_chip_io_writes = 0
    buffer_space = 0
    buffer_nodes_space = []
    streaming_ios = 0

    for src, dst, data in dag.edges(data=True):

        # Look at all edges that are non streaming or that arrive to a buffer node
        if 'stream' not in data or data['stream'] == False or dst in buffer_nodes:

            volume = data['weight']

            # if this is a buffer node and it is mapped in the on_chip memory we need to count
            # the data movements as on-chip ones
            if src in buffer_nodes_on_chip:
                on_chip_io_reads += volume
            else:
                off_chip_io_reads += volume

            if dst in buffer_nodes_on_chip:
                on_chip_io_writes += volume
            else:
                off_chip_io_writes += volume
        else:
            # this is a streaming edge
            streaming_ios += 2 * volume  # writes and read

        # Sanity checks:
        #  Edges departing from buffer nodes are
        #  always non-streaming (as by construction of the canonical DAG)
        if src in buffer_nodes or src in pseudo_nodes:
            assert 'stream' not in data or data['stream'] == False

    for bn in buffer_nodes:
        # get buffer space by looking at input edges
        # TODO: if this is data coming from the memory anyway(input data of the program), should we count as well?
        # I would say no. Further refinement may consider to store this on-chip for fast access

        if dag.out_degree(bn) == 0:  #source or  sink node
            continue

        # if (dag.in_degree(bn) == 1 and [n for n in dag.predecessors(bn)][0] in pseudo_nodes):
        #     print("Node ", bn, "has only one predecessor and it is a pseudo node", [n for n in dag.predecessors(bn)][0])
        # if (dag.out_degree(bn) == 1 and [n for n in dag.successors(bn)][0] in pseudo_nodes):
        #     print("Node ", bn, "has only one successor and it is a pseudo node", [n for n in dag.successors(n)(bn)][0])

        if (dag.in_degree(bn) == 1 and [n for n in dag.predecessors(bn)][0]
                in pseudo_nodes) or (dag.out_degree(bn) == 1 and [n for n in dag.successors(bn)][0] in pseudo_nodes):
            # The only predecessor (successor) is the pseudo source (sink) node. This data is anyway in memoery
            continue

        buff_data = get_buffer_node_space(dag, bn)
        buffer_space += data["weight"]
        buffer_nodes_space.append(data["weight"])

    # print(buffer_nodes_space)
    return off_chip_io_reads + off_chip_io_writes, on_chip_io_reads + on_chip_io_writes, streaming_ios, buffer_space, np.histogram(
        buffer_nodes_space, bins=10)


def count_io_canonical_dag_local(dag: nx.DiGraph, buffer_nodes: set, pseudo_nodes: set, spatial_blocks: list):
    '''
        Given a Canonical DAG and its partitioning in spatial blocks, returns statistics on the I/O
        buffer space usage.

        For each meaningful spatial block (i.e. that does not include only pseudo and buffer nodes) this counts:
        - the buffer space counted for buffer node in the block (or for buffer nodes in adjacent blocks that will be read? -- this will be counted in the prev. block)
        - the number of I/Os
        - np.histogram for I/Os occuring between blocks

        These are returned as aggregate, min, max, median
        OR SOMETHING ELSE?
    '''

    # TODO: double check everything

    buffer_space_per_block = []
    ios_per_block = []

    ios_btw_blocks = []
    ios_block = 0
    for sb in spatial_blocks:

        buff = 0
        ios = 0

        #TODO: maybe do it with subgraph?

        # Look at just the nodes in the block
        for node in sb:
            # if this node has non-streaming edges, these are coming from a previous block (or from a buffer node)
            for src, dst, data in dag.in_edges(node, data=True):
                if 'stream' not in data or data['stream'] is False:
                    ios += data['weight']
                    if src not in buffer_nodes and src not in sb:  # or not in sb? the buffer node can still be in a different block
                        ios_block += data[
                            'weight']  # note that a buffer node may still be in the same block but its outgoing edges are non-stremaing

            if node in buffer_nodes:
                # TODO: this assumes the weight is always the same

                buff += data['weight']

            # then have a look also to outgoing edges, but only the ones that go out of the block
            for src, dst, data in dag.out_edges(node, data=True):
                if dst not in sb and ('stream' not in data or data['stream'] is False):
                    ios += data['weight']
                    if dst not in buffer_nodes and dst not in sb:
                        ios_block += data['weight']

        # print("Block: ", sb)
        # print("\t IOs: ", ios)
        # print("\t buffer space: ", buff)

        buffer_space_per_block.append(buff)
        ios_per_block.append(ios)
        ios_btw_blocks.append(ios_block)

    buff_space = (min(buffer_space_per_block), statistics.median(buffer_space_per_block), max(buffer_space_per_block))
    ios = (min(ios_per_block), statistics.median(ios_per_block), max(ios_per_block))
    # print("Stats: ", buff_space, ios)
    return buff_space, ios, np.histogram(ios_btw_blocks, bins=10)


def knapsack(V, wt, val):
    """
    Solves knapsack problem given volume V, weights wt and corresponding values values.
    Returns the ids (position in the weights/value arrays) of the items that
    must be included in the knapsack

    :param V: _description_
    :type V: _type_
    :param wt: _description_
    :type wt: _type_
    :param val: _description_
    :type val: _type_
    :param n: _description_
    :type n: _type_
    """
    n = len(val)
    K = [[0 for w in range(V + 1)] for i in range(n + 1)]

    # Build table K[][] in bottom  up manner
    for i in range(n + 1):
        for w in range(V + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = K[n][V]

    w = V
    results = set()
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # either the result comes from the top (K[i-1][w]) or from (val[i-1]
        # + K[i-1] [w-wt[i-1]]) as in Knapsack table. If it comes from the latter
        # one/ it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:

            # This item is included.
            # print(wt[i - 1], i)
            results.add(i - 1)

            # Since this weight is included
            # its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]
    return results