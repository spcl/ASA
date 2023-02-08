# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.

# Note: this test is used for analysis purposes (we rely on DaCeML pure expansion for this operation)

## Needs DaCe 0.13.3

from dace.transformation.interstate import InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from daceml.torch import DaceModule, dace_module
import dace
import argparse
from daceml.util import utils
from dace.transformation.interstate import InlineSDFG


class Model(nn.Module):
    def __init__(self, start, stop):
        super(Model, self).__init__()
        self.start = start
        self.stop = stop

    def forward(self, x):
        x = x[self.start:self.stop, :]
        return x


def run(data_shape: tuple, start: int, stop: int, queue=None):
    '''
    Evaluates a specific configuration
    '''
    ptmodel = Model(start, stop)
    x = torch.rand(data_shape)

    torch_output = ptmodel(torch.clone(x))
    import daceml.onnx as donnx
    with dace.library.change_default(donnx.ONNXSlice, "pure"):
        dace_model = DaceModule(
            ptmodel,
            auto_optimize=False,
            dummy_inputs=(x, ),
        )
        # dace_output = dace_model(x)
    # assert np.allclose(torch_output.detach().numpy(), dace_output)

    sdfg = dace_model.sdfg
    sdfg.view()
    from dace.transformation.dataflow import RedundantArray
    sdfg.apply_transformations_repeated([RedundantArray], print_report=True)


if __name__ == "__main__":
    run((96, 32), 0, 32)