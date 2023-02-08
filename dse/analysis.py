# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
Collection of analysis tools for the results produced by DSE
"""

import numpy as np
from dse.utils import save_results_to_file


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def get_pareto_configurations(results: dict, sdfg_name: str = "sdfg"):
    """
    Computes the pareto frontier of the given results.

    Currently it consider three costs:
    - Dynamic and Static Powers scores
    - Makespan (performance)
    - Area score 

    For these three, the lower, the better

    TODO: have a system to plug in different cost models

    :param results: results as dict num_pes -> list of results
    """

    # Build An (n_points, n_costs) array
    n_points = 0
    previous = 0
    for k, v in results.items():
        n_points += len(v)
        if previous == 0:
            previous = len(v)
        else:
            assert len(
                v
            ) == previous  # all num of pes have the same number of results (used later to print the pareto frontier)

    cost_array = np.zeros([n_points, 4]).astype(np.int64)  # array: dynamic power, static power, perf, area

    i = 0
    ids = []  # keep track of the various configurations
    for k, v in results.items():
        for idx, r in enumerate(v):
            # TODO: create infrastructure for cost models
            cost_array[i, 0] = r.dynamic_power
            cost_array[i, 1] = r.static_power
            cost_array[i, 2] = r.performance
            cost_array[i, 3] = r.area
            ids.append((k, idx))
            # TODO: deal with buffer space
            # if r.buffer_space > 0:
            #     print("Configuration with buffer space > 0", k, r)
            i += 1

    # compute pareto frontier
    is_efficient = is_pareto_efficient(cost_array)
    pareto_frontier = []
    for i, is_e in enumerate(is_efficient):
        if is_e:
            # print(f"Point {i} is on the pareto frontier {cost_array[i]}")

            num_pes = ids[i][0]
            pos = ids[i][1]
            r = results[num_pes][pos]
            # print("\t", r)
            pareto_frontier.append([
                num_pes, r.makespan, r.off_chip_IOs, r.on_chip_IOs, r.streaming_IOs, r.dynamic_power, r.static_power,
                r.performance, r.area, r.number_of_nodes, r.number_of_buffer_nodes, r.time_expansion, r.time_inlining,
                r.time_scheduling, r.ns_makespan, r.buffer_space_deadlock, r.buffer_space,
                list(r.buffer_nodes_space_histogram[0]), [round(item) for item in r.buffer_nodes_space_histogram[1]],
                r.number_of_blocks, r.buffer_space_per_block, r.ios_per_block,
                list(r.ios_btw_blocks[0]), [round(item) for item in r.ios_btw_blocks[1]], r.num_pes, r.on_chip_space,
                r.expansion_list
            ])

    ## save pareto frontier  to file
    # TODO: create script to do this from already stored csv results

    if r.num_iterations > 1 or r.unroll_factor > 1:
        results_filename = f'pareto_{sdfg_name}_nit_{r.num_iterations}_unrolling_{r.unroll_factor}.csv'
    else:
        results_filename = f'pareto_{sdfg_name}.csv'

    results_header = [
        "Num_Pes", "Makespan", "Off-Chip IOs", "On-Chip IOs", "Streaming IOs", "Dynamic Power Score",
        "Static Power Score", "Performance score", "Area score", "Total number of canonical nodes",
        "Number of buffer nodes", "SDFG expansion time", "SDFG inlining time", "Scheduling time",
        "Non-Streaming Makespan", "Buffer space for deadlock prev.", "IOs distribution", "IOs bins",
        "Total Buffer space buff nodes", "Buffer Nodes space distribution", "Buffer Nodes Space Bins",
        "Number of blocks", "Buff Space per block", "IOs per block", "IOs btw blocks space distribution",
        "IOs btw blocks Space Bins", "Number of PEs", "On-Chip Memory", "Expansion list"
    ]

    save_results_to_file(results_filename, results_header, pareto_frontier)

    return


def ppa_model_base(makespan: int, num_pes: int, off_chip_IOs: int, on_chip_IOs: int, on_chip_memory: int):
    """
    First simplistic model for PPA estimation.
    Returns a score as estimate of Power (Dynamic and Static), Performance and Area 
    """

    # Power estimates:  the power consumption is given by:
    #   - dynamic power: a combination of off-chip and on-chip IOs. Off-chip accesses are more power consuming
    #   - static power: the number of PEs
    alpha = 0.8  # off-chip weight
    beta = 0.2  # on-chip weight
    gamma = 0.5  # PEs wait

    dynamic_power = alpha * off_chip_IOs + beta * on_chip_IOs
    static_power = gamma * num_pes

    # Performance estimate: we just use the makespan
    performance = makespan

    # Area estimates: depends on the amount of used on chip memoery and number of PEs
    alpha = 0.5  # weight for on-chip memory
    beta = 1  # wight for num_pes

    area = alpha * on_chip_memory + beta * num_pes

    return dynamic_power, static_power, performance, area
