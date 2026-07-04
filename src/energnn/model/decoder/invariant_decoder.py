# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from energnn.graph import ADDRESS_GRAPH_IDS, Graph
from energnn.model.utils import MLP, scatter_add
from .decoder import Decoder


def _sum_over_addresses(graph: Graph, h: jax.Array) -> jax.Array:
    """Sum per-address values into a global vector, or per-instance vectors for a union graph.

    For a regular graph, returns ``sum(h, axis=0)`` of shape ``(d,)``. For a disjoint-union
    graph (with segment metadata), returns per-instance sums of shape ``(n_graphs, d)``;
    fictitious addresses carry an out-of-range id and are dropped by the scatter.
    """
    if graph.segments is not None:
        accumulator = jnp.zeros((graph.n_union_graphs, h.shape[-1]))
        return scatter_add(accumulator=accumulator, increment=h, addresses=graph.segments[ADDRESS_GRAPH_IDS])
    return jnp.sum(h, axis=0)


class InvariantDecoder(Decoder, ABC):
    """Abstract base class for invariant decoders that produce global outputs.

    Invariant decoders aggregate information from all addresses in a permutation-invariant
    manner to produce a single global output vector.
    """

    @abstractmethod
    def __call__(self, *, graph: Graph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Decode latent coordinates into a global decision vector.

        :param graph: Input graph to decode.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple containing decision vector and info dictionary.
        :raises NotImplementedError: If subclass does not override this method.
        """
        raise NotImplementedError


class SumInvariantDecoder(InvariantDecoder):
    r"""
    Sum invariant decoder, that sums the information of all addresses.

    .. math::
        \hat{y} = \phi_\theta \left( \sum_{a \in \mathcal{A}(x)} \psi_\theta(h_a)\right),

    where :math:`\phi_\theta` (outer) and :math:`\psi_\theta` (inner) are both trainable MLPs.

    :param psi: Inner MLP :math:`\psi_\theta`.
    :param phi: Outer MLP :math:`\phi_\theta`.
    """

    def __init__(self, *, psi: MLP, phi: MLP) -> None:
        super().__init__()
        self.psi = psi
        self.phi = phi

    def __call__(self, *, graph: Graph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        h = self.psi(coordinates)
        h = h * jnp.expand_dims(graph.non_fictitious_addresses, -1)
        h = _sum_over_addresses(graph, h)
        out = self.phi(h)
        return out, {}


class MeanInvariantDecoder(InvariantDecoder):
    r"""
    Mean invariant decoder, that averages the information of all addresses.

    .. math::
        \hat{y} = \phi_\theta \left( \frac{1}{\vert \mathcal{A}(x) \vert} \sum_{a \in \mathcal{A}(x)} \psi_\theta(h_a) \right),

    where :math:`\phi_\theta` (outer) and :math:`\psi_\theta` (inner) are both trainable MLPs.

    :param psi: Inner MLP :math:`\psi_\theta`.
    :param phi: Outer MLP :math:`\phi_\theta`.
    """

    def __init__(self, *, psi: MLP, phi: MLP) -> None:
        super().__init__()
        self.psi = psi
        self.phi = phi

    def __call__(self, *, graph: Graph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        numerator = self.psi(coordinates)
        numerator = numerator * jnp.expand_dims(graph.non_fictitious_addresses, -1)
        numerator = _sum_over_addresses(graph, numerator)
        denominator = _sum_over_addresses(graph, jnp.expand_dims(graph.non_fictitious_addresses, -1)) + 1e-9
        return self.phi(numerator / denominator), {}
