# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Throughput benchmark matrix for the compute/batching/parallelism options.

Sweeps {dense | union} x {fp32 | bf16} x {per-port | fused} on the local devices
and reports steady-state ms/step plus a short convergence check. Designed to be
run on a GPU/TPU host, where bf16 (tensor cores) and MLP fusion (batched GEMM,
fewer kernel launches) are expected to help — and where union batching removes
the padding waste that dominates heterogeneous-size dense batches.

Single device::

    python benchmarks/parallel_gpu_matrix.py

Multiple devices on one host (data-parallel over per-device disjoint unions)::

    python benchmarks/parallel_gpu_matrix.py --shard

Multiple hosts: launch one process per host with the same flags after
``jax.distributed.initialize()`` (see docs/parallelism.md); the matrix uses
``jax.process_count()``/``process_index()`` automatically when ``--shard`` and a
multi-process cluster are detected.
"""

from __future__ import annotations

import argparse
import itertools
import time

import jax
import numpy as np
import optax

from energnn.model.ready_to_use import ReadyRecurrentEquivariantGNN
from energnn.problem.example import LinearSystemProblemLoader
from energnn.trainer import Trainer


def _time_train(model_kwargs, loader_kwargs, mesh, n_warmup_epochs, n_timed_epochs, lr):
    train_loader = LinearSystemProblemLoader(seed=1, **loader_kwargs)
    val_loader = LinearSystemProblemLoader(seed=2, **loader_kwargs)
    model = ReadyRecurrentEquivariantGNN(
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        n_breakpoints=20,
        latent_dimension=16,
        hidden_sizes=[32],
        n_steps=20,
        normalizer_external_updates=True,  # required for multi-host, harmless otherwise
        **model_kwargs,
    )
    trainer = Trainer(model=model, gradient_transformation=optax.adam(lr), mesh=mesh)
    score_before, _ = trainer.eval(val_loader, get_info=False)

    trainer.train(train_loader=train_loader, n_epochs=n_warmup_epochs, progress_bar=False)  # compile
    n_steps = n_timed_epochs * len(train_loader)
    start = time.perf_counter()
    trainer.train(train_loader=train_loader, n_epochs=n_timed_epochs, progress_bar=False)
    ms_per_step = (time.perf_counter() - start) / max(n_steps, 1) * 1000

    score_after, _ = trainer.eval(val_loader, get_info=False)
    return ms_per_step, score_before, score_after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="store_true", help="Data-parallel over all local/global devices.")
    parser.add_argument("--n-max", type=int, default=64, help="Max nodes per instance (size heterogeneity).")
    parser.add_argument("--batch-size", type=int, default=64, help="Global batch size (instances per batch).")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--timed-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    n_devices = jax.device_count()
    mesh = None
    if args.shard:
        from jax.sharding import Mesh

        mesh = Mesh(np.array(jax.devices()), ("data",))

    print(
        f"devices={n_devices} process_count={jax.process_count()} shard={args.shard} "
        f"n_max={args.n_max} batch_size={args.batch_size}"
    )
    print(f"{'batching':>8} {'dtype':>8} {'mlps':>8} {'ms/step':>10} {'score':>22}")

    modes = ["dense", "union"] if not args.shard else ["union"]
    for batching, dtype, fused in itertools.product(modes, ["fp32", "bf16"], [False, True]):
        loader_kwargs = dict(dataset_size=args.batch_size, batch_size=args.batch_size, n_max=args.n_max, mode=batching)
        if args.shard:
            loader_kwargs.update(
                n_shards=n_devices,
                process_count=jax.process_count(),
                process_index=jax.process_index(),
                mesh=mesh,
            )
        model_kwargs = dict(
            compute_dtype="bfloat16" if dtype == "bf16" else None,
            fuse_port_mlps=fused,
        )
        try:
            ms, s0, s1 = _time_train(model_kwargs, loader_kwargs, mesh, args.warmup_epochs, args.timed_epochs, args.lr)
            label = "fused" if fused else "per-port"
            if jax.process_index() == 0:
                print(f"{batching:>8} {dtype:>8} {label:>8} {ms:>10.2f} {f'{s0:.4f}->{s1:.4f}':>22}")
        except Exception as exc:  # keep the sweep going if one config fails
            if jax.process_index() == 0:
                label = "fused" if fused else "per-port"
                print(f"{batching:>8} {dtype:>8} {label:>8} {'FAILED':>10} {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
