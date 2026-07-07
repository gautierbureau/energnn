# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""The GPU-oriented compute flags (bf16, fused port MLPs) compose with each other
and with union batching. Speedups are GPU-only, but correctness/composition is
validated here on CPU."""

import numpy as np

from energnn.model.ready_to_use import ReadyRecurrentEquivariantGNN
from energnn.problem.example import LinearSystemProblemLoader
from energnn.problem.example.linear_system import LinearSystemProblemGenerator


def _model(loader, **kwargs):
    return ReadyRecurrentEquivariantGNN(
        in_structure=loader.context_structure,
        out_structure=loader.decision_structure,
        n_breakpoints=20,
        latent_dimension=8,
        hidden_sizes=[16],
        n_steps=6,
        seed=0,
        normalizer_external_updates=True,
        **kwargs,
    )


def _forward(model, context):
    model.eval()
    out, _ = model.forward_batch(graph=context, get_info=False)
    return np.array(out.hyper_edge_sets["bus"].feature_array)


def test_fused_port_mlps_match_per_port():
    """Fusing the port MLPs is numerically equivalent to applying them one by one."""
    loader = LinearSystemProblemLoader(seed=1, n_max=12)
    context = LinearSystemProblemGenerator(seed=9, n_max=12).generate_problem_batch(batch_size=4, mode="union").context
    per_port = _forward(_model(loader), context)
    fused = _forward(_model(loader, fuse_port_mlps=True), context)
    np.testing.assert_allclose(per_port, fused, rtol=1e-5, atol=1e-6)


def test_bf16_and_fusion_compose():
    """bf16 and fusion compose: the fused bf16 path matches the per-port bf16 path,
    and bf16 stays close to full precision."""
    loader = LinearSystemProblemLoader(seed=1, n_max=12)
    context = LinearSystemProblemGenerator(seed=9, n_max=12).generate_problem_batch(batch_size=4, mode="union").context
    fp32 = _forward(_model(loader), context)
    bf16_per_port = _forward(_model(loader, compute_dtype="bfloat16"), context)
    bf16_fused = _forward(_model(loader, compute_dtype="bfloat16", fuse_port_mlps=True), context)

    # The dtype-aware fused path is equivalent to the per-port path at the same precision.
    np.testing.assert_allclose(bf16_per_port, bf16_fused, rtol=1e-3, atol=1e-3)
    # bf16 stays close to full precision (bf16 has an 8-bit mantissa).
    rel = np.max(np.abs(fp32 - bf16_per_port)) / (np.max(np.abs(fp32)) + 1e-9)
    assert rel < 0.06, rel
