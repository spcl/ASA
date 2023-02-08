# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Canonical expansion for MaxPooling
'''
import dace
from dace.transformation.transformation import ExpandTransformation
from canonical_libnodes.utils import in_desc_with_name, out_desc_with_name
from dace import properties
import copy


@dace.library.expansion
class ExpandMaxPoolCanonical(ExpandTransformation):
    '''
    Canonical expansion for MaxPool

    '''

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_sdfg")

        # Get descriptors (note this may have a shape that is not the one where we expect. E.g., a view on a higher order container
        # will show here the original shape of the container, not the viewed one. In that case use something like _get_mamul_operator)
        X = in_desc_with_name(node, parent_state, parent_sdfg, "X")
        Y = out_desc_with_name(node, parent_state, parent_sdfg, "Y")

        # Checks
        image_dims = len(X.shape) - 2
        # only do 2D for now
        assert image_dims == 2
        assert node.strides == node.kernel_shape

        batch_size = X.shape[0]
        num_channels = X.shape[1]
        input_size_height = X.shape[2]
        input_size_width = X.shape[3]
        output_size_height = Y.shape[2]
        output_size_width = Y.shape[3]
        filter_height, filter_width = node.kernel_shape

        state = sdfg.add_state(node.label + "_state")
        sdfg.add_datadesc("X", copy.deepcopy(X))
        sdfg.add_datadesc("Y", copy.deepcopy(Y))
        sdfg.arrays["X"].transient = False
        sdfg.arrays["Y"].transient = False

        ## Outer map loops over all output entries
        outer_me, outer_mx = state.add_map("output_map", {
            "b": f"0:{batch_size}",
            "c": f"0:{num_channels}",
            "h": f"0:{output_size_height}",
            "w": f"0:{output_size_width}",
        },
                                           schedule=dace.ScheduleType.Sequential)

        ## Inside the map we compute the max with an inner map
        # Create temp variable and initialize it
        sdfg.add_scalar("max_value", Y.dtype, storage=dace.StorageType.Register, transient=True)
        max_access_in = state.add_access("max_value")
        max_access_out = state.add_access("max_value")

        init_max = state.add_tasklet("init_max", {}, {"max_out"},
                                     f"max_out = {dace.dtypes.min_value(Y.dtype.base_type)}")

        state.add_memlet_path(outer_me, init_max, memlet=dace.Memlet())
        state.add_memlet_path(init_max, max_access_in, src_conn="max_out", memlet=dace.Memlet("max_value[0]"))

        ## Inner map to compute the pool
        inner_me, inner_mx = state.add_map('inner_pool_map', {
            "hx": f"0:{filter_height}",
            "hy": f"0:{filter_width}"
        },
                                           schedule=dace.ScheduleType.Sequential)

        # Tasklet to compute the max
        compute_tasklet = state.add_tasklet("compute_max",
                                            inputs={"image_in", "max_in"},
                                            outputs={"max_out"},
                                            code="max_out = float(max(max_in, image_in))")

        # Connect
        x_read = state.add_read("X")
        y_write = state.add_write("Y")
        state.add_memlet_path(max_access_in,
                              inner_me,
                              compute_tasklet,
                              dst_conn="max_in",
                              memlet=dace.Memlet("max_value[0]"))
        state.add_memlet_path(x_read,
                              outer_me,
                              inner_me,
                              compute_tasklet,
                              dst_conn="image_in",
                              memlet=dace.Memlet(f"X[b, c, h * {filter_height} + hx, w * {filter_width} + hy]"))
        state.add_memlet_path(compute_tasklet,
                              inner_mx,
                              max_access_out,
                              src_conn="max_out",
                              memlet=dace.Memlet("max_value[0]"))
        state.add_memlet_path(max_access_out, outer_mx, y_write, memlet=dace.Memlet("Y[b,c,h,w]"))
        return sdfg

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        return ExpandMaxPoolCanonical.make_sdfg(node, state, sdfg)
