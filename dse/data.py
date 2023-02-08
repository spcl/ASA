# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
Data class definition
"""
from dataclasses import dataclass


@dataclass
class Result():
    '''

    '''
    expansion_list: list
    num_pes: int  # number of PEs being used
    on_chip_space: int  # size of on-chip memory area
    makespan: int  # the computation time of the application (already taking into account multiple iterations if any)
    off_chip_IOs: int  # data movements from off-chip memory: either buffer nodes in external memory or non-streaming edges
    on_chip_IOs: int  # data movements to/from buffer nodes that reside in on-chip memory
    streaming_IOs: int  # data movements along streaming edges and computational nodes
    performance: float  # performance score
    dynamic_power: float  # dynamic power score
    static_power: float  # static power score
    area: float  # area score
    time_expansion: float
    ns_makespan: int = 0  # non streaming makespan (0 if not scheduled)
    unroll_factor: int = 1  # if the application is iterated, this tells how many time has been unrolled while computing the scheduling
    num_iterations: int = 1  # tells how many time the schedule must be iterated in the case of iterative algorithm (+ partially unrolled) # TODO: recompute automatically the schedule
    time_inlining: float = 0
    time_scheduling: float = 0
    number_of_nodes: int = 0  # number of nodes in the canonical Task
    number_of_buffer_nodes: int = 0
    buffer_space_deadlock: int = 0  # buffer space to prevent deadlock
    buffer_space: int = 0  # buffer space coming from buffer nodes
    buffer_nodes_space_histogram: tuple = (
        [0], [0])  # histogram of buffer_space for buffer node distribution (by default in 10 bins)
    number_of_blocks: int = 0  # approx number of blocks (some of them may have only buffer nodes)
    buffer_space_per_block: tuple = (0, 0, 0)
    ios_per_block: tuple = (0, 0, 0)  # all the IOs of a block (considering also buffer node)
    ios_btw_blocks: tuple = ([0], [0])  # histogram of ios between blocks (not counting buffer nodes)
