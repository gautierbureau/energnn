# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import subprocess
import sys

import numpy as np

from energnn.graph import collate_graphs, separate_graphs
from energnn.problem.example.linear_system import LinearSystemProblemGenerator

# Multi-device behavior needs XLA_FLAGS set before JAX initializes, so the
# device-mesh test runs in a subprocess with a faked 4-device CPU platform.
_MULTI_DEVICE_SCRIPT = """
import jax, jax.numpy as jnp, numpy as np, optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from energnn.problem.example import LinearSystemProblemLoader
from energnn.problem.example.linear_system import LinearSystemProblemGenerator
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from energnn.trainer import Trainer

assert len(jax.devices()) == 4
mesh = Mesh(np.array(jax.devices()), ("data",))

# Sharded batched-union forward matches the dense path per instance.
loader = LinearSystemProblemLoader(seed=1, n_max=8)
model = TinyRecurrentEquivariantGNN(in_structure=loader.context_structure, out_structure=loader.decision_structure)
model.eval()
dense = LinearSystemProblemGenerator(seed=9, n_max=8).generate_problem_batch(batch_size=8, mode="dense")
shard = LinearSystemProblemGenerator(seed=9, n_max=8).generate_problem_batch(batch_size=8, mode="union", n_shards=4)
forward = nnx.jit(lambda m, g: m.forward_batch(graph=g, get_info=False)[0])
out_dense = forward(model, dense.context)
context_sharded = jax.device_put(shard.context, NamedSharding(mesh, P("data")))
out_sharded = forward(model, context_sharded)
dense_features = np.array(out_dense.hyper_edge_sets["bus"].feature_array)
sharded_features = np.array(out_sharded.hyper_edge_sets["bus"].feature_array)
true_bus = np.array(shard.context.segments["true_shapes"].hyper_edge_sets["bus"])
instance = 0
for d in range(true_bus.shape[0]):
    offset = 0
    for k in range(true_bus.shape[1]):
        size = int(true_bus[d, k])
        if size == 0:
            continue
        np.testing.assert_allclose(
            sharded_features[d, offset:offset + size], dense_features[instance, :size], rtol=2e-5, atol=1e-6)
        offset += size
        instance += 1
assert instance == 8

# Message passing over per-device unions requires no cross-device communication.
hlo = nnx.jit(lambda m, g: m.forward_batch(graph=g, get_info=False)[0]).lower(model, context_sharded).compile().as_text()
n_collectives = sum(hlo.count(x) for x in ("all-reduce", "all-gather", "all-to-all", "collective-permute"))
assert n_collectives == 0, f"expected no collectives in the forward, found {n_collectives}"

# Mesh-sharded training produces the same parameters as single-device training.
def train(mesh_arg):
    tl = LinearSystemProblemLoader(seed=1, dataset_size=16, batch_size=8, n_max=8, mode="union", n_shards=4)
    m = TinyRecurrentEquivariantGNN(in_structure=tl.context_structure, out_structure=tl.decision_structure)
    tr = Trainer(model=m, gradient_transformation=optax.adam(1e-2), mesh=mesh_arg)
    tr.train(train_loader=tl, n_epochs=2, progress_bar=False)
    return m

single, sharded = train(None), train(mesh)
for a, b in zip(jax.tree.leaves(nnx.state(single, nnx.Param)), jax.tree.leaves(nnx.state(sharded, nnx.Param))):
    np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-6)
print("MULTI_DEVICE_OK")
"""


def test_data_parallel_union_training_multi_device():
    """Sharded batched-union forward/training matches the single-device results on a faked 4-device mesh."""
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", _MULTI_DEVICE_SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "MULTI_DEVICE_OK" in result.stdout


def test_collate_and_separate_preserve_segments():
    """Batching unions with collate_graphs keeps segment metadata, and separate_graphs restores it."""
    generator = LinearSystemProblemGenerator(seed=0, n_max=8)
    batch = generator.generate_problem_batch(batch_size=6, mode="union", n_shards=3)

    assert batch.context.segments is not None
    assert batch.context.is_batch
    ids = np.array(batch.context.segments["hyper_edge_graph_ids"]["bus"])
    assert ids.ndim == 2 and ids.shape[0] == 3

    separated = separate_graphs(batch.context)
    assert len(separated) == 3
    for i, union in enumerate(separated):
        assert union.segments is not None
        np.testing.assert_array_equal(np.array(union.segments["hyper_edge_graph_ids"]["bus"]), ids[i])

    rebatched = collate_graphs(separated)
    np.testing.assert_array_equal(np.array(rebatched.segments["hyper_edge_graph_ids"]["bus"]), ids)


def test_sharded_scores_preserve_instance_order():
    """Per-instance scores from a sharded batch come back in generation order, padded slots dropped."""
    generator = LinearSystemProblemGenerator(seed=3, n_max=8)
    sharded = generator.generate_problem_batch(batch_size=7, mode="union", n_shards=3)
    reference = LinearSystemProblemGenerator(seed=3, n_max=8).generate_problem_batch(batch_size=7, mode="union")

    decision_sharded, _ = sharded.get_zero_decision()
    decision_reference, _ = reference.get_zero_decision()
    scores_sharded, _ = sharded.get_score(decision=decision_sharded)
    scores_reference, _ = reference.get_score(decision=decision_reference)
    assert len(scores_sharded) == 7
    np.testing.assert_allclose(scores_sharded, scores_reference, rtol=1e-6)
