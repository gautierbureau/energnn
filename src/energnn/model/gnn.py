# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax
from flax import nnx

from energnn.graph import Graph
from .coupler import Coupler
from .decoder import Decoder
from .encoder import Encoder
from .normalizer import Normalizer


class GNN(nnx.Module):
    """
    Simple Graph Neural Network (GNN) model designed to handle Hyper Heterogeneous Multi Graphs (H2MGs).

    The model consists of a normalization step, an encoding step, a coupling step, and a decoding step.
    The decoder can either be invariant or equivariant, depending on the task requirements.

    :param normalizer: Maps the input features to a learning-compatible range.
    :type normalizer: Normalizer
    :param encoder: Embeds hyper-edge set features into a latent space.
    :type encoder: Encoder
    :param coupler: Outputs latent coordinates for each address present in the input graph.
    :type coupler: Coupler
    :param decoder: Maps latent coordinates and encoded graph to a meaningful output.
    :type decoder: Decoder
    """

    def __init__(self, normalizer: Normalizer, encoder: Encoder, coupler: Coupler, decoder: Decoder):
        self.normalizer = normalizer
        self.encoder = encoder
        self.coupler = coupler
        self.decoder = decoder

    def __call__(self, graph: Graph, get_info: bool = False) -> tuple[Graph | jax.Array, dict]:
        """
        Processes a given graph through a sequence of steps: normalization, encoding, coupling,
        and decoding. The method applies a series of transformations to the input graph and
        returns a decoded graph / array along with optional processing information.

        :param graph: The input graph to be processed.
        :param get_info: A boolean indicating whether detailed processing information should
            be returned. Defaults to False.
        :return: A tuple consisting of the processed decoded graph / array and an optional dictionary
            with detailed information about each processing step if `get_info` is True.
        """
        info = {}
        normalized_graph, info["normalization"] = self.normalizer(graph=graph, get_info=get_info)
        encoded_graph, info["encoding"] = self.encoder(graph=normalized_graph, get_info=get_info)
        latent_coordinates, info["coupling"] = self.coupler(graph=encoded_graph, get_info=get_info)
        output, info["decoding"] = self.decoder(coordinates=latent_coordinates, graph=encoded_graph, get_info=get_info)
        return output, info

    def ingest(self, *, graph: Graph) -> None:
        """Update the normalizer's statistics from a context graph, outside the jitted forward.

        No-op unless the normalizer is configured for external updates
        (see :meth:`energnn.model.normalizer.Normalizer.ingest`). The trainer calls this
        once per training step on the context batch, before the forward pass.

        :param graph: Context graph whose features should be ingested.
        """
        ingest = getattr(self.normalizer, "ingest", None)
        if ingest is not None:
            ingest(graph=graph)

    def forward_batch(self, *, graph: Graph, get_info: bool = False) -> tuple[Graph | jax.Array, dict]:
        """Applies the model to a batch of graphs.

        Dense batches (built with :func:`~energnn.graph.collate_graphs`) are processed by
        vmapping the encoder, coupler, and decoder over the batch axis, while the
        normalization module is not vmapped. Disjoint-union batches (built with
        :func:`~energnn.graph.union_graphs`) are single larger graphs and go through the
        plain forward pass, which scales with the sum of instance sizes instead of
        ``batch * max_size``. A *batched* graph of unions (one union per device, built
        with ``collate_graphs`` over unions) takes the vmapped path, with each lane
        processing one union — the layout used for multi-device data parallelism.

        :param graph: Batch of input graphs.
        :param get_info: Whether to return additional information about the processing steps.
        """
        if graph.segments is not None and graph.is_single:
            return self(graph=graph, get_info=get_info)

        def apply_core(encoder, coupler, decoder, graph, get_info):
            info = {}
            encoded_graph, info["encoding"] = encoder(graph=graph, get_info=get_info)
            latent_coordinates, info["coupling"] = coupler(graph=encoded_graph, get_info=get_info)
            output, info["decoding"] = decoder(coordinates=latent_coordinates, graph=encoded_graph, get_info=get_info)
            return output, info

        normalized_graph, info_norm = self.normalizer(graph=graph, get_info=get_info)
        output, info_core = jax.vmap(apply_core, in_axes=[None, None, None, 0, None], out_axes=0)(
            self.encoder, self.coupler, self.decoder, normalized_graph, get_info
        )
        return output, info_norm | info_core
