# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.typing import Dtype, Initializer

from energnn.graph import GraphStructure, Graph
from energnn.model.utils import Activation, MLP, gather, scatter_add


class MessagePassingFunction(nnx.Module, ABC):
    r"""Interface for a message function :math:`\xi_\theta` in a GNN message passing scheme."""

    @abstractmethod
    def __call__(self, graph: Graph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Should take as input a tuple (graph, coordinates) and return new coordinates."""
        raise NotImplementedError


def _can_fuse_mlps(mlps: list) -> bool:
    """True if all modules are standard MLPs with identical architecture.

    Such MLPs can be applied to the same input as a single batched pass over their
    stacked weights, instead of one chain of small matmuls per port.
    """
    if len(mlps) < 2:
        return False
    if not all(type(m) is MLP for m in mlps):
        return False
    first = mlps[0]
    return all(
        m.in_size == first.in_size
        and m.hidden_sizes == first.hidden_sizes
        and m.out_size == first.out_size
        and m.use_bias == first.use_bias
        and m.activation is first.activation
        and m.final_activation is first.final_activation
        and m.dtype == first.dtype
        for m in mlps[1:]
    )


def _fused_mlp_apply(mlps: list[MLP], x: jax.Array, dtype=None) -> jax.Array:
    """Apply identically-shaped MLPs to the same input in one batched pass.

    Stacks the per-MLP layer weights along a leading axis and contracts each layer
    with a single einsum, so ``len(mlps)`` port-specific MLPs cost one batched matmul
    per layer instead of one small matmul each. When ``dtype`` is set the matmuls run
    in that dtype (mirroring ``nnx.Linear(dtype=...)``), so the fused path composes with
    bf16 mixed precision.

    :param mlps: MLPs with identical architecture (see :func:`_can_fuse_mlps`).
    :param x: Common input array of shape ``(..., in_size)``.
    :param dtype: Compute dtype for the matmuls, or None for full precision.
    :return: Stacked outputs of shape ``(len(mlps), ..., out_size)``.
    """
    y = x
    first_linear = True
    for i, layer in enumerate(mlps[0].sequential.layers):
        if isinstance(layer, nnx.Linear):
            kernel = jnp.stack([m.sequential.layers[i].kernel[...] for m in mlps])
            if dtype is not None:
                y = y.astype(dtype)
                kernel = kernel.astype(dtype)
            if first_linear:
                y = jnp.einsum("...f,pfo->p...o", y, kernel)
                first_linear = False
            else:
                y = jnp.einsum("p...f,pfo->p...o", y, kernel)
            if layer.bias is not None:
                bias = jnp.stack([m.sequential.layers[i].bias[...] for m in mlps])
                if dtype is not None:
                    bias = bias.astype(dtype)
                y = y + bias.reshape(bias.shape[0], *([1] * (y.ndim - 2)), bias.shape[1])
        else:
            y = layer(y)
    return y


class LocalSumMessagePassingFunction(MessagePassingFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to hyper-edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    For each address :math:`a`, the output is defined as:

    .. math::
        \psi_\theta(h,x)_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_x(a)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, :math:`\sigma` is an
    element-wise activation function, and :math:`h_e := (h_{o(e)})_{o \in {\mathcal{O}^c}}` is the concatenation of
    port coordinates of hyper-edge :math:`e`.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs :math:`\xi^{c,o}_\theta`.
    :param activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param out_size: Output size of the MLPs :math:`\xi^{c,o}_\theta`.
    :param use_bias: Whether to use bias in the MLPs :math:`\xi^{c,o}_\theta`.
    :param kernel_init: Kernel initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param bias_init: Bias initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param final_activation: Final activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param outer_activation: Activation function :math:`\sigma` applied over the output.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param port_scatter_blacklist: Dictionary mapping hyper-edge set keys to lists of port keys to be excluded from the sum.
    :param dtype: Computation dtype of the MLPs :math:`\xi^{c,o}_\theta` (e.g. ``jnp.bfloat16``
        for mixed precision); parameters stay float32 and the scatter accumulation runs in
        float32. None (default) computes in full precision.
    :param fuse_port_mlps: If True, apply a class's port MLPs (identical architecture) as a
        single batched einsum over their stacked weights instead of one small matmul per
        port. Numerically equivalent; a throughput trade-off that tends to help on GPU
        (batched GEMM, fewer kernel launches) and hurt on CPU. Default False. Modules that
        are not standard MLPs fall back to the per-port path.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        dtype: Dtype | None = None,
        fuse_port_mlps: bool = False,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.encoded_feature_size = encoded_feature_size
        self.dtype = dtype
        self.fuse_port_mlps = fuse_port_mlps
        if port_scatter_blacklist is None:
            self.port_scatter_blacklist = {}
        else:
            self.port_scatter_blacklist = port_scatter_blacklist

        self.mlp_tree = self._build_mlp_tree(seed=seed, rngs=rngs)

    def _build_mlp_tree(self, seed: int = 0, rngs: nnx.Rngs | None = None) -> dict[str, dict[str, MLP]]:
        if rngs is None:
            rngs = nnx.Rngs(seed)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")
        mlp_tree = {}

        for key, hyper_edge_set_structure in self.in_graph_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.port_list is not None and len(hyper_edge_set_structure.port_list) > 0:
                n_ports = len(hyper_edge_set_structure.port_list)
                in_size = self.in_array_size * n_ports
                if hyper_edge_set_structure.feature_list is not None and len(hyper_edge_set_structure.feature_list) > 0:
                    if self.encoded_feature_size is not None:
                        in_size += self.encoded_feature_size
                    else:
                        in_size += len(hyper_edge_set_structure.feature_list)

                if key not in mlp_tree.keys():
                    mlp_tree[key] = {}

                for port_key in hyper_edge_set_structure.port_list:
                    if port_key not in self.port_scatter_blacklist.get(key, []):
                        mlp_tree[key][port_key] = MLP(
                            in_size=in_size,
                            hidden_sizes=self.hidden_sizes,
                            activation=self.activation,
                            out_size=self.out_size,
                            use_bias=self.use_bias,
                            kernel_init=self.kernel_init,
                            bias_init=self.bias_init,
                            final_activation=self.final_activation,
                            dtype=self.dtype,
                            rngs=rngs,
                        )
        return nnx.data(mlp_tree)

    def __call__(self, *, graph: Graph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        accumulator = jnp.zeros((coordinates.shape[0], self.out_size))

        # Iterate classes and ports in sorted key order (a stable, backend-independent order).
        for key in sorted(graph.hyper_edge_sets.keys()):
            hyper_edge_set = graph.hyper_edge_sets[key]
            mlp_dict = self.mlp_tree[key]

            input_array = []
            if hyper_edge_set.feature_names is not None:
                input_array.append(hyper_edge_set.feature_array)
            for port_name, port_array in hyper_edge_set.port_dict.items():
                input_array.append(gather(coordinates=coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array, axis=-1)
            non_fictitious_mask = jnp.expand_dims(hyper_edge_set.non_fictitious, -1)
            masked_input_array = input_array * non_fictitious_mask

            port_items = sorted(mlp_dict.items(), key=lambda kv: kv[0])
            mlps = [mlp for _, mlp in port_items]

            if self.fuse_port_mlps and _can_fuse_mlps(mlps):
                # All port MLPs share one architecture: run them as one batched pass.
                outputs = _fused_mlp_apply(mlps, masked_input_array, dtype=self.dtype)
                for (port_name, _), output in zip(port_items, outputs):
                    increment = output * non_fictitious_mask
                    accumulator = scatter_add(
                        accumulator=accumulator, increment=increment, addresses=hyper_edge_set.port_dict[port_name]
                    )
            else:
                for port_name, mlp in port_items:
                    increment = mlp(masked_input_array) * non_fictitious_mask
                    accumulator = scatter_add(
                        accumulator=accumulator, increment=increment, addresses=hyper_edge_set.port_dict[port_name]
                    )

        return self.outer_activation(accumulator), {}


class IdentityMessagePassingFunction(MessagePassingFunction):
    r"""
    Identity local message function module for GNN message passing.

    This module returns the node features unchanged as the local message.
    It implements the identity mapping on node features:

    .. math::
        h^\rightarrow_a = h_a
    """

    def __init__(self):
        pass

    def __call__(self, *, graph: Graph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}
