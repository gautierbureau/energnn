# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

from flax import nnx

from energnn.graph import Graph


class Normalizer(nnx.Module, ABC):
    """Interface for a normalizer.

    A normalizer transforms the input graph features into a distribution
    more suitable for neural network training (e.g., standardization, normalization).
    """

    @abstractmethod
    def __call__(self, graph: Graph, get_info: bool = False) -> tuple[Graph, dict]:
        """Normalize the input graph features.

        :param graph: Input graph to normalize.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: A tuple containing:
            - Normalized graph with transformed features
            - A dictionary with additional information if get_info=True, empty dict otherwise
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def ingest(self, *, graph: Graph) -> None:
        """Update the normalizer's statistics from a graph, outside of the jitted forward pass.

        Default is a no-op. Normalizers whose statistics updates would otherwise require a
        host round-trip inside the compiled forward (e.g. T-Digest ingestion) can override
        this and be configured to only update through explicit ``ingest`` calls, keeping
        their ``__call__`` free of host callbacks. The :class:`~energnn.trainer.Trainer`
        calls this once per training step on the context batch, before the forward pass.

        :param graph: Graph whose features should be ingested.
        """
