# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Collection of utils for library node creation (borrowed from DaCeML)
'''

from dace import SDFG, SDFGState, dtypes
from dace import nodes as nd
import dace.data as dt
from dace.sdfg.state import MultiConnectorEdge


def in_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """

    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG, name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def in_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to input connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the input connector name.
        :return: the edge that connects to connector `name`.
     """

    cands = list(state.in_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: nd.Node, state: SDFGState, name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to output connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the output connector name.
        :return: the edge that connects to connector `name`.
     """
    cands = list(state.out_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError("Expected to find exactly one edge with name '{}', found {}".format(name, len(cands)))
    return cands[0]