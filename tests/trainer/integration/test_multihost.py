# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import subprocess
import sys

import numpy as np
import pytest

from energnn.graph import separate_graphs
from energnn.problem.example.linear_system import LinearSystemProblemGenerator

# A genuine 2-process cluster runs in a pair of subprocesses with a faked CPU
# platform; each host owns 2 of 4 global devices.
_TWO_PROCESS_WORKER = """
import os, sys
proc = int(sys.argv[1])
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import jax
jax.distributed.initialize(coordinator_address="127.0.0.1:{port}", num_processes=2, process_id=proc)
import numpy as np, optax
from flax import nnx
from jax.sharding import Mesh
from energnn.problem.example import LinearSystemProblemLoader
from energnn.model.ready_to_use import ReadyRecurrentEquivariantGNN
from energnn.trainer import Trainer

assert jax.process_count() == 2 and jax.device_count() == 4 and jax.local_device_count() == 2
mesh = Mesh(np.array(jax.devices()), ("data",))
# Keep the workload tiny: one batch per epoch (dataset_size == batch_size) with a
# fixed n_max so all batches share one padded shape -> a single SPMD compilation.
common = dict(dataset_size=4, batch_size=4, n_max=4, mode="union", n_shards=4,
              process_count=2, process_index=jax.process_index(), mesh=mesh)
tl = LinearSystemProblemLoader(seed=1, **common)
vl = LinearSystemProblemLoader(seed=2, **common)
# Multi-host requires a callback-free forward, so the normalizer updates on the host.
model = ReadyRecurrentEquivariantGNN(
    in_structure=tl.context_structure, out_structure=tl.decision_structure,
    n_breakpoints=10, latent_dimension=4, hidden_sizes=[], n_steps=5,
    normalizer_external_updates=True,
)
trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-2), mesh=mesh)
score_before, _ = trainer.eval(vl, get_info=False)
trainer.train(train_loader=tl, n_epochs=3, progress_bar=False)
score_after, _ = trainer.eval(vl, get_info=False)
# Parameters and the evaluation score are globally reduced, so every process agrees.
param_sum = float(np.asarray(jax.tree.leaves(nnx.state(model, nnx.Param))[0]).sum())
assert score_after < score_before, (score_before, score_after)
print(f"OK proc={{jax.process_index()}} paramsum={{param_sum:.6f}} score={{score_after:.6f}}")
jax.distributed.shutdown()
"""


def test_process_partitioning_reconstructs_global_batch():
    """Each process's shards, concatenated, reproduce the single-process global batch.

    This is the host-side partitioning logic (no cross-process assembly), fully
    exercisable on one machine.
    """
    full = LinearSystemProblemGenerator(seed=7, n_max=8).generate_problem_batch(batch_size=8, mode="union", n_shards=4)
    p0 = LinearSystemProblemGenerator(seed=7, n_max=8).generate_problem_batch(
        batch_size=8, mode="union", n_shards=4, process_count=2, process_index=0
    )
    p1 = LinearSystemProblemGenerator(seed=7, n_max=8).generate_problem_batch(
        batch_size=8, mode="union", n_shards=4, process_count=2, process_index=1
    )
    full_unions = separate_graphs(full.context)
    local_unions = separate_graphs(p0.context) + separate_graphs(p1.context)
    assert len(full_unions) == 4 and len(local_unions) == 4
    for expected, actual in zip(full_unions, local_unions):
        np.testing.assert_allclose(
            np.array(expected.hyper_edge_sets["bus"].feature_array),
            np.array(actual.hyper_edge_sets["bus"].feature_array),
            rtol=1e-6,
        )


def test_host_callback_normalizer_guard():
    """A host-callback normalizer (default T-Digest) is rejected for genuine multi-host meshes."""
    from energnn.model.ready_to_use import ReadyRecurrentEquivariantGNN, TinyRecurrentEquivariantGNN
    from energnn.problem.example import LinearSystemProblemLoader
    from energnn.trainer.trainer import _check_no_host_callbacks

    loader = LinearSystemProblemLoader(seed=0)
    default_model = TinyRecurrentEquivariantGNN(in_structure=loader.context_structure, out_structure=loader.decision_structure)
    with pytest.raises(ValueError, match="external_updates"):
        _check_no_host_callbacks(default_model)

    external_model = ReadyRecurrentEquivariantGNN(
        in_structure=loader.context_structure,
        out_structure=loader.decision_structure,
        n_breakpoints=10,
        latent_dimension=4,
        hidden_sizes=[],
        n_steps=5,
        normalizer_external_updates=True,
    )
    _check_no_host_callbacks(external_model)  # must not raise


def test_generator_rejects_bad_process_layout():
    generator = LinearSystemProblemGenerator(seed=0, n_max=8)
    # process_count must divide n_shards
    try:
        generator.generate_problem_batch(batch_size=8, mode="union", n_shards=4, process_count=3, process_index=0)
        assert False, "expected ValueError for indivisible n_shards"
    except ValueError:
        pass
    # multi-process requires union mode
    try:
        generator.generate_problem_batch(batch_size=8, mode="dense", process_count=2, process_index=0)
        assert False, "expected ValueError for dense multi-process"
    except ValueError:
        pass


@pytest.mark.skipif(
    os.environ.get("ENERGNN_RUN_MULTIHOST_TEST") != "1",
    reason="Spawns a 2-process JAX cluster (slow on CPU); set ENERGNN_RUN_MULTIHOST_TEST=1 to run.",
)
def test_two_process_data_parallel_training():
    """A real 2-process cluster trains, agrees on globally-reduced params, and converges."""
    port = 12500 + (os.getpid() % 500)
    worker = os.path.join(os.path.dirname(__file__), "_multihost_worker_tmp.py")
    with open(worker, "w") as handle:
        handle.write(_TWO_PROCESS_WORKER.format(port=port))
    try:
        clean_env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
        procs = [
            subprocess.Popen(
                [sys.executable, worker, str(rank)], env=clean_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            for rank in range(2)
        ]
        outputs = [p.communicate(timeout=600)[0].decode() for p in procs]
    finally:
        if os.path.exists(worker):
            os.remove(worker)
    for rank, (proc, out) in enumerate(zip(procs, outputs)):
        assert proc.returncode == 0, f"process {rank} failed:\n{out}"
        assert "OK proc=" in out, f"process {rank} did not report success:\n{out}"
    # Both processes must report the same globally-reduced parameter sum.
    param_sums = [
        float(line.split("paramsum=")[1].split()[0]) for out in outputs for line in out.splitlines() if "paramsum=" in line
    ]
    assert abs(param_sums[0] - param_sums[1]) < 1e-4, param_sums
