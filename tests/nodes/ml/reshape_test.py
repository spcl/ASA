## Needs DaCe 0.13.3
# DaCEML pure expansion should be already canonical

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import onnx
import numpy as np
import pytest
from daceml.torch import DaceModule, dace_module
import dace
import argparse
from daceml.util import utils
from dace.transformation.interstate import InlineSDFG


class Model(nn.Module):
    def __init__(self, new_shape):
        super(Model, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x.reshape(self.new_shape)
        return x


def run(data_shape: tuple, reshaped_shape: tuple, vec_width=1, queue=None):

    ptmodel = Model(reshaped_shape)
    x = torch.rand(data_shape)

    torch_output = ptmodel(x)

    import daceml.onnx as donnx
    with dace.library.change_default(donnx.ONNXReshape, "pure"):
        dace_model = DaceModule(ptmodel, auto_optimize=False, dummy_inputs=(x, ))
        out = dace_model(x)
    sdfg = dace_model.sdfg
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    sdfg.view()
    assert np.allclose(out, torch_output.detach().numpy())


if __name__ == "__main__":

    data_shape = (16, 4, 4, 4)
    reshaped_shape = (16, 64)
    run(data_shape, reshaped_shape)
