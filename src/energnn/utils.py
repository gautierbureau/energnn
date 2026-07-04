# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os

import jax


def enable_persistent_compilation_cache(cache_dir: str | None = None, min_compile_time_secs: float = 1.0) -> str:
    """
    Enable JAX's persistent compilation cache, so XLA compilations survive process restarts.

    Without this, every new process pays the full compilation cost of the model again
    (several seconds to minutes for deep couplers) before the first training or
    evaluation step runs. With the cache enabled, compiled executables are stored on
    disk and re-loaded on subsequent runs with the same model, shapes, and JAX version.

    Call this once, before the first JIT compilation (ideally right after importing JAX).

    .. note::
        JAX cannot persist executables that contain host callbacks. In particular, the
        *training-mode* forward pass of a model using :class:`TDigestNormalizer` embeds
        an ``io_callback`` and is recompiled in every process; the backward pass, the
        optimizer update, and evaluation-mode forwards are cached normally.

    :param cache_dir: Directory used to store compiled executables. Created if missing.
        Defaults to ``~/.cache/energnn/jax_compilation_cache``.
    :param min_compile_time_secs: Only compilations slower than this are persisted,
        so the cache is not cluttered with trivially recompilable kernels.
    :return: The cache directory in use.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "energnn", "jax_compilation_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    return cache_dir
