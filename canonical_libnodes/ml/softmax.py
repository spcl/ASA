# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
"""
    Canonical expansion for Softmax operator.
    This is a composite expansions that uses other canonical library nodes.
"""

import dace
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties
import copy
import daceml


@dace.library.expansion
class ExpandSoftmaxSum(ExpandTransformation):
    '''
    Computes the denominators for a Softmax
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        inparr = in_desc_with_name(node, parent_state, parent_sdfg, "input")
        outarr = out_desc_with_name(node, parent_state, parent_sdfg, "sum_data_bn")

        #### Add containers
        _, array_input = sdfg.add_array("input", inparr.shape, dtype=inparr.dtype)
        _, array_output = sdfg.add_array("sum_data_bn", outarr.shape, dtype=outarr.dtype)

        # Add container to store sums result
        sdfg.add_scalar("partial_sum", dtype=inparr.dtype.base_type, transient=True)

        partial_sum = state.add_access("partial_sum")
        partial_sum_out = state.add_access("partial_sum")
        input_read_for_sum = state.add_read("input")

        ### Compute the denominators (sum of exp elements) along all dimension except axis
        # TODO: represent this as separate Reduction node
        # TODO: remove this and just use map in the future

        map_ranges = {'__i%d' % i: '0:%s' % n for i, n in enumerate(inparr.shape[:-1])}

        # all the other axis map
        other_sum_me, other_sum_mx = state.add_map("softmax_map", map_ranges)

        # sum map
        sum_me, sum_mx = state.add_map("softmax_sum", dict(i=f"0:{inparr.shape[-1]}"))

        # init sum container
        init_sum = state.add_tasklet("init_sum", {}, {"sum_out"}, f"sum_out = 0")

        state.add_memlet_path(other_sum_me, init_sum, memlet=dace.Memlet())
        state.add_memlet_path(init_sum, partial_sum, src_conn="sum_out", memlet=dace.Memlet("partial_sum[0]"))

        # compute sum
        sum_tasklet = state.add_tasklet('sum_task', {'_in', '_in_sum'}, {'_out_sum'}, "_out_sum = _in_sum + exp(_in)")

        memlet_except_axis = f"{','.join(['__i%d' % i for i in range(len(inparr.shape) - 1)])}"

        state.add_memlet_path(input_read_for_sum,
                              other_sum_me,
                              sum_me,
                              sum_tasklet,
                              dst_conn="_in",
                              memlet=dace.Memlet(f"input[{memlet_except_axis},i]"))

        state.add_memlet_path(partial_sum,
                              sum_me,
                              sum_tasklet,
                              dst_conn="_in_sum",
                              memlet=dace.Memlet(f"partial_sum[0]"))
        state.add_memlet_path(sum_tasklet,
                              sum_mx,
                              partial_sum_out,
                              src_conn="_out_sum",
                              memlet=dace.Memlet(f"partial_sum[0]"))
        # save sum to sum_data_bn
        sum_data_access = state.add_access("sum_data_bn")
        state.add_memlet_path(partial_sum_out,
                              other_sum_mx,
                              sum_data_access,
                              memlet=dace.Memlet(f"sum_data_bn[{memlet_except_axis}]"))
        return sdfg


@dace.library.node
class SofmaxSum(dace.sdfg.nodes.LibraryNode):
    '''
        Adds the C, row by row in the computed matrix
        
    '''

    # Global properties
    implementations = {"default": ExpandSoftmaxSum}
    default_implementation = "default"

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs={"input"}, outputs={"sum_data_bn"})

    def validate(self, sdfg, state):
        #TODO
        pass


@dace.library.expansion
class ExpandSoftmaxDiv(ExpandTransformation):
    '''
    Computes the division (using sums) producing the final result
    '''

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        node.validate(parent_sdfg, parent_state)
        sdfg = dace.SDFG(node.label + "_sdfg")
        state = sdfg.add_state(node.label + "_state")
        inparr = in_desc_with_name(node, parent_state, parent_sdfg, "input")
        sumarr = in_desc_with_name(node, parent_state, parent_sdfg, "sum_data_bn")
        outarr = out_desc_with_name(node, parent_state, parent_sdfg, "output")

        #### Add containers
        _, array_input = sdfg.add_array("input", inparr.shape, dtype=inparr.dtype)
        _, array_sum = sdfg.add_array("sum_data_bn", sumarr.shape, dtype=sumarr.dtype)
        _, array_output = sdfg.add_array("output", outarr.shape, dtype=outarr.dtype)

        ### Compute the denominators (sum of exp elements) along all dimension except axis
        # TODO: represent this as separate Reduction node
        # TODO: remove this and just use map in the future

        map_ranges = {'__i%d' % i: '0:%s' % n for i, n in enumerate(inparr.shape[:-1])}
        memlet_except_axis = f"{','.join(['__i%d' % i for i in range(len(inparr.shape) - 1)])}"

        input_read_for_exp = state.add_read("input")
        output_write = state.add_write("output")
        sum_data_access = state.add_read("sum_data_bn")

        # all the other axis map
        other_exp_me, other_exp_mx = state.add_map("softmax_div", map_ranges)

        # exp map
        exp_div_me, exp_div_mx = state.add_map("softmax_exp_div", dict(i=f"0:{inparr.shape[-1]}"))

        # compute exp and div
        exp_div_tasklet = state.add_tasklet('exp_task', {'_in', '_in_sum'}, {'_out'}, "_out = exp(_in)/_in_sum")

        state.add_memlet_path(input_read_for_exp,
                              other_exp_me,
                              exp_div_me,
                              exp_div_tasklet,
                              dst_conn="_in",
                              memlet=dace.Memlet(f"input[{memlet_except_axis},i]"))

        state.add_memlet_path(sum_data_access,
                              other_exp_me,
                              exp_div_me,
                              exp_div_tasklet,
                              dst_conn="_in_sum",
                              memlet=dace.Memlet(f"sum_data_bn[{memlet_except_axis}]"))

        # save result
        state.add_memlet_path(exp_div_tasklet,
                              exp_div_mx,
                              other_exp_mx,
                              output_write,
                              src_conn="_out",
                              memlet=dace.Memlet(f"output[{memlet_except_axis},i]"))
        return sdfg


@dace.library.node
class SofmaxDiv(dace.sdfg.nodes.LibraryNode):
    '''
        Computes the final result
        
    '''

    # Global properties
    implementations = {"default": ExpandSoftmaxDiv}
    default_implementation = "default"

    def __init__(self, name, location=None):
        super().__init__(name, location=location, inputs={"input", "sum_data_bn"}, outputs={"output"})

    def validate(self, sdfg, state):
        #TODO
        pass


@dace.library.expansion
class ExpandSoftmaxCanonical(ExpandTransformation):
    '''
    Canonical expansion for Softmax
    This is the unstable version, computing exp(x_i)/sum(exp(x_i))


    TODO: numberically stable version


    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")

        inparr = in_desc_with_name(node, parent_state, parent_sdfg, "input")
        outarr = out_desc_with_name(node, parent_state, parent_sdfg, "output")

        axis = node.axis
        if type(axis) is not int or not (-len(inparr.shape) <= axis < len(inparr.shape)):
            raise ValueError(
                f"expected axis to be an integer in range [-{len(inparr.shape)}, {len(inparr.shape)}), got {axis}")

        # TODO: not sure about this. We would need to change map ranges in case
        assert axis == len(inparr.shape) - 1

        if axis < 0:
            axis += len(inparr.shape)

        state = sdfg.add_state(node.label + "_state")
        sdfg.add_datadesc("input", copy.deepcopy(inparr))
        sdfg.add_datadesc("output", copy.deepcopy(outarr))
        sdfg.arrays["input"].transient = False
        sdfg.arrays["output"].transient = False

        # Add container to store sums result
        _, sum_data_bn_array = sdfg.add_array("sum_data_bn",
                                              inparr.shape[0:axis],
                                              dtype=inparr.dtype.base_type,
                                              transient=True)

        input_read_for_sum = state.add_read("input")
        sum_data_access = state.add_access("sum_data_bn")
        map_ranges = {'__i%d' % i: '0:%s' % n for i, n in enumerate(inparr.shape[:-1])}
        memlet_except_axis = f"{','.join(['__i%d' % i for i in range(len(inparr.shape) - 1)])}"

        ### Compute the denominators (sum of exp elements) along all dimension except axis
        # TODO: represent this as separate Reduction node
        # TODO: remove this and just use map in the future

        libnode = SofmaxSum('SoftmaxSum')

        state.add_node(libnode)

        state.add_edge(input_read_for_sum, None, libnode, 'input',
                       dace.Memlet.from_array("input", sdfg.arrays["input"]))

        state.add_edge(libnode, 'sum_data_bn', sum_data_access, None,
                       dace.Memlet.from_array("sum_data_bn", sum_data_bn_array))

        ### Exp map
        input_read_for_exp = state.add_read("input")
        output_write = state.add_write("output")

        libnode = SofmaxDiv('SoftmaxDiv')

        state.add_node(libnode)

        state.add_edge(input_read_for_exp, None, libnode, 'input',
                       dace.Memlet.from_array("input", sdfg.arrays["input"]))

        state.add_edge(sum_data_access, None, libnode, 'sum_data_bn',
                       dace.Memlet.from_array("sum_data_bn", sum_data_bn_array))

        state.add_edge(libnode, 'output', output_write, None, dace.Memlet.from_array("output", sdfg.arrays["output"]))

        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandSoftmaxCanonical.make_sdfg(node, state, sdfg)