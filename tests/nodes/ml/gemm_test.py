# ASA-DSE: Application Specific Architecture Toolchain.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.

## Needs DaCe v0.13.3

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pytest
from daceml.torch import DaceModule, dace_module

import dace
import daceml

from canonical_libnodes.ml.gemm import ExpandGemmCanonical


class Model(nn.Module):
    def __init__(self, in_features=120, out_features=84):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


def run(batch_size=10, input_features=120, output_features=84):
    '''
    Evaluates the given configuration
    :param batch_size, input_features, output_features: data size
    :return:
    '''

    x = torch.rand(batch_size, input_features, dtype=torch.float32)
    # build the DaCe model from the pytorch model
    ptmodel = Model(in_features=input_features, out_features=output_features)

    torch_output = ptmodel(x)
    import daceml.onnx as donnx

    daceml.onnx.nodes.onnx_op.ONNXGemm.register_implementation("canonical", ExpandGemmCanonical)
    with dace.library.change_default(donnx.ONNXGemm, "canonical"):
        dace_model = DaceModule(ptmodel, dummy_inputs=(x, ), auto_optimize=False)
        dace_output = dace_model(x)

    # This is needed in case you want to view the SDFG and/or convert it to canonical DAG
    sdfg = dace_model.sdfg
    gemm_node = sdfg.nodes()[0].nodes()[0]
    gemm_node.implementation = "canonical"
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
    diff = np.linalg.norm(torch_output.detach().numpy() - dace_output.numpy()) / np.linalg.norm(
        torch_output.detach().numpy())
    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)


def gemm_test():
    run()


if __name__ == "__main__":
    run()
