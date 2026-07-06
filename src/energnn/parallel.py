# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Helpers for multi-host (multi-process) data-parallel training.

These wrap the JAX multi-controller idioms used to run one training program per
process, each driving its local devices, over a shared global device mesh. Every
helper degrades to a plain single-process operation when ``jax.process_count() == 1``,
so the same code path runs unchanged on a single host and can be validated there.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.experimental import multihost_utils


def data_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    """Sharding that splits an array along its leading (batch) axis over ``mesh``."""
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(mesh.axis_names[0]))


def replicated_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    """Sharding that replicates an array on every device of ``mesh``."""
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def assemble_global(pytree, sharding: jax.sharding.NamedSharding):
    """
    Assemble a pytree of process-local host arrays into global ``jax.Array`` s.

    Each leaf is treated as this process's contiguous slice along the leading axis;
    the global leading dimension is ``local_leading_dim * jax.process_count()``.
    On a single process this is exactly equivalent to ``jax.device_put(pytree, sharding)``,
    so the multi-host path is exercised (and validated) even with one process.

    :param pytree: Pytree of process-local arrays (all leaves batched on axis 0).
    :param sharding: Leading-axis sharding over the global mesh (see :func:`data_sharding`).
    :return: Pytree of global ``jax.Array`` s placed according to ``sharding``.
    """
    process_count = jax.process_count()

    def _assemble_leaf(x):
        local = np.asarray(x)
        global_shape = (local.shape[0] * process_count,) + local.shape[1:]
        return jax.make_array_from_process_local_data(sharding, local, global_shape)

    return jax.tree.map(_assemble_leaf, pytree)


def replicate(pytree, mesh: jax.sharding.Mesh):
    """Place a pytree as fully-replicated global arrays on ``mesh`` (e.g. model parameters).

    Each process supplies its own (identical) copy of every leaf. Uses
    ``make_array_from_process_local_data`` rather than ``jax.device_put`` to a replicated
    sharding, because the latter runs a cross-process equality assert that nan-initialized
    state (e.g. the T-Digest min/max) would fail even when the copies are identical.
    """
    sharding = replicated_sharding(mesh)

    def _replicate_leaf(x):
        local = np.asarray(x)
        return jax.make_array_from_process_local_data(sharding, local, local.shape)

    return jax.tree.map(_replicate_leaf, pytree)


def gather_to_host(array) -> np.ndarray:
    """
    Materialize a (possibly globally-sharded) array as a replicated host NumPy array.

    Concatenates the shards from all processes along the leading axis. On a single
    process this is just ``np.asarray``. Needed before host-side reductions of
    per-instance results (e.g. validation scores) that are sharded across processes.
    """
    if jax.process_count() > 1:
        return np.asarray(multihost_utils.process_allgather(array, tiled=True))
    return np.asarray(array)
