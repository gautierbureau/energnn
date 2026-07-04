# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import pickle as pkl

import numpy as np
from jax.tree_util import register_pytree_node_class

from energnn.graph.backend import Backend, NumpyBackend
from energnn.graph.hyper_edge_set import (
    HyperEdgeSet,
    collate_hyper_edge_sets,
    concatenate_hyper_edge_sets,
    separate_hyper_edge_sets,
)
from energnn.graph.shape import GraphShape, collate_shapes, separate_shapes, sum_shapes

HYPER_EDGE_SETS = "hyper_edge_sets"
TRUE_SHAPE = "true_shape"
CURRENT_SHAPE = "current_shape"
NON_FICTITIOUS_ADDRESSES = "non_fictitious_addresses"
SEGMENTS = "segments"

# Keys of the optional ``segments`` metadata carried by disjoint-union graphs.
ADDRESS_GRAPH_IDS = "address_graph_ids"
HYPER_EDGE_GRAPH_IDS = "hyper_edge_graph_ids"
TRUE_SHAPES = "true_shapes"


@register_pytree_node_class
class Graph(dict):
    """
    Hyper Heterogeneous Multi-Graph (H2MG) container.

    Stores hyper-edge sets, shapes, and address masks for single or batched graphs.
    All array operations are delegated to the provided *backend*.

    :param backend: Array backend (:class:`NumpyBackend` or :class:`JaxBackend`).
    :param hyper_edge_sets: Dict of hyper-edge sets.
    :param true_shape: True shape, unaffected by padding.
    :param current_shape: Current shape, consistent with padding.
    :param non_fictitious_addresses: 1 for real addresses, 0 otherwise.
    :param segments: Optional disjoint-union metadata created by :func:`union_graphs`
        (per-address and per-hyper-edge instance ids plus per-instance true shapes).
        ``None`` for regular single or batched graphs.
    """

    def __init__(
        self,
        *,
        backend: Backend | None = None,
        hyper_edge_sets: dict[str, HyperEdgeSet],
        true_shape: GraphShape,
        current_shape: GraphShape,
        non_fictitious_addresses,
        segments: dict | None = None,
    ) -> None:
        super().__init__()
        self._backend: Backend = backend if backend is not None else NumpyBackend()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[TRUE_SHAPE] = true_shape
        self[CURRENT_SHAPE] = current_shape
        self[NON_FICTITIOUS_ADDRESSES] = non_fictitious_addresses
        self[SEGMENTS] = segments

    # ------------------------------------------------------------------
    # JAX PyTree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = list(self.values())
        aux = (tuple(self.keys()), self._backend)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children) -> Graph:
        keys, backend = aux_data
        d = dict(zip(keys, children))
        return cls(
            backend=backend,
            hyper_edge_sets=d[HYPER_EDGE_SETS],
            true_shape=d[TRUE_SHAPE],
            current_shape=d[CURRENT_SHAPE],
            non_fictitious_addresses=d[NON_FICTITIOUS_ADDRESSES],
            segments=d.get(SEGMENTS),
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, *, hyper_edge_set_dict: dict[str, HyperEdgeSet], n_addresses, backend: Backend | None = None) -> Graph:
        """
        Build a Graph from a dict of HyperEdgeSet and a number of addresses.

        :param hyper_edge_set_dict: Dict of hyper-edge sets.
        :param n_addresses: Number of unique addresses.
        :param backend: Array backend; defaults to NumpyBackend (or inferred from hyper_edge_sets).
        :return: New Graph.
        """
        if backend is None:
            if hyper_edge_set_dict:
                backend = next(iter(hyper_edge_set_dict.values()))._backend
            else:
                backend = NumpyBackend()
        xp = backend.xp

        non_fictitious_addresses = xp.ones(shape=[int(n_addresses)])
        check_hyper_edge_set_dict_type(hyper_edge_set_dict)
        check_valid_addresses(hyper_edge_set_dict, n_addresses)
        true_shape = GraphShape.from_dict(
            hyper_edge_set_dict=hyper_edge_set_dict, non_fictitious=non_fictitious_addresses, backend=backend
        )
        return cls(
            backend=backend,
            hyper_edge_sets=hyper_edge_set_dict,
            true_shape=true_shape,
            current_shape=true_shape,
            non_fictitious_addresses=non_fictitious_addresses,
        )

    # ------------------------------------------------------------------
    # Backend conversion
    # ------------------------------------------------------------------

    def to_backend(self, new_backend: Backend) -> Graph:
        """Return a copy of this Graph with all arrays converted to ``new_backend``."""
        hyper_edge_sets_b = {k: hes.to_backend(new_backend) for k, hes in self.hyper_edge_sets.items()}
        true_shape_b = self.true_shape.to_backend(new_backend)
        current_shape_b = self.current_shape.to_backend(new_backend)
        nfa_b = new_backend.from_numpy(np.array(self.non_fictitious_addresses))
        segments_b = None
        if self.segments is not None:
            segments_b = {
                ADDRESS_GRAPH_IDS: new_backend.from_numpy(np.array(self.segments[ADDRESS_GRAPH_IDS]), dtype="int32"),
                HYPER_EDGE_GRAPH_IDS: {
                    k: new_backend.from_numpy(np.array(v), dtype="int32")
                    for k, v in self.segments[HYPER_EDGE_GRAPH_IDS].items()
                },
                TRUE_SHAPES: self.segments[TRUE_SHAPES].to_backend(new_backend),
            }
        return type(self)(
            backend=new_backend,
            hyper_edge_sets=hyper_edge_sets_b,
            true_shape=true_shape_b,
            current_shape=current_shape_b,
            non_fictitious_addresses=nfa_b,
            segments=segments_b,
        )

    @classmethod
    def to_jax_backend(cls, graph: Graph, device=None, dtype: str = "float32") -> Graph:
        """Convert a NumPy-backed Graph to a JAX-backed one."""
        from energnn.graph.backend import JaxBackend

        return graph.to_backend(JaxBackend(device=device, dtype=dtype))

    def to_numpy_backend(self) -> Graph:
        """Convert this Graph to a NumPy-backed one."""
        return self.to_backend(NumpyBackend())

    # ------------------------------------------------------------------
    # Pickle
    # ------------------------------------------------------------------

    def to_pickle(self, file_path: str) -> None:
        """Save this graph to a pickle file."""
        with open(file_path, "wb") as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, *, file_path: str) -> Graph:
        """Load a graph from a pickle file."""
        with open(file_path, "rb") as handle:
            return pkl.load(handle)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def true_shape(self) -> GraphShape:
        return self[TRUE_SHAPE]

    @property
    def current_shape(self) -> GraphShape:
        return self[CURRENT_SHAPE]

    @current_shape.setter
    def current_shape(self, value: GraphShape) -> None:
        self[CURRENT_SHAPE] = value

    @property
    def non_fictitious_addresses(self):
        return self[NON_FICTITIOUS_ADDRESSES]

    @non_fictitious_addresses.setter
    def non_fictitious_addresses(self, value) -> None:
        self[NON_FICTITIOUS_ADDRESSES] = value

    @property
    def segments(self) -> dict | None:
        """Disjoint-union metadata (see :func:`union_graphs`); None for regular graphs."""
        return self.get(SEGMENTS)

    @segments.setter
    def segments(self, value: dict | None) -> None:
        self[SEGMENTS] = value

    @property
    def n_union_graphs(self) -> int:
        """Number of instances in a disjoint-union graph (including padded empty instances).

        :raises ValueError: If this graph carries no union segment metadata.
        """
        if self.segments is None:
            raise ValueError("This graph is not a disjoint union (no segment metadata).")
        return int(self.segments[TRUE_SHAPES].addresses.shape[0])

    @property
    def hyper_edge_sets(self) -> dict[str, HyperEdgeSet]:
        return self[HYPER_EDGE_SETS]

    @hyper_edge_sets.setter
    def hyper_edge_sets(self, hyper_edge_set_dict: dict[str, HyperEdgeSet]) -> None:
        self[HYPER_EDGE_SETS] = hyper_edge_set_dict

    def __str__(self) -> str:
        return "".join("{}\n{}\n".format(k, v) for k, v in sorted(self.hyper_edge_sets.items()))

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    @property
    def is_batch(self) -> bool:
        """True if all hyper-edge sets are batched and the address mask is 2-D."""
        for e in self.hyper_edge_sets.values():
            if not e.is_batch:
                return False
        if self.non_fictitious_addresses is not None and len(self.non_fictitious_addresses.shape) != 2:
            return False
        return True

    @property
    def is_single(self) -> bool:
        """True if all hyper-edge sets are single and the address mask is 1-D."""
        for e in self.hyper_edge_sets.values():
            if not e.is_single:
                return False
        if self.non_fictitious_addresses is not None and len(self.non_fictitious_addresses.shape) != 1:
            return False
        return True

    # ------------------------------------------------------------------
    # Feature flat array
    # ------------------------------------------------------------------

    @property
    def feature_flat_array(self):
        """Concatenated flat features of all hyper-edge sets."""
        xp = self._backend.xp
        values_list = []
        if self.hyper_edge_sets is not None:
            for key, hes in sorted(self.hyper_edge_sets.items()):
                if hes.feature_flat_array is not None:
                    values_list.append(hes.feature_flat_array)
        else:
            raise ValueError("This graph does not contain any hyper-edge set, and can't be cast as a flat array.")
        return xp.concatenate(values_list, axis=-1)

    @feature_flat_array.setter
    def feature_flat_array(self, value) -> None:
        if self.feature_flat_array.shape != value.shape:
            raise ValueError("Invalid array shape.")
        i = 0
        if self.hyper_edge_sets is not None:
            for key, hes in sorted(self.hyper_edge_sets.items()):
                if hes.feature_names is not None:
                    length = hes.feature_flat_array.shape[-1]
                    if length > 0:
                        self.hyper_edge_sets[key].feature_flat_array = value[..., i : i + length]
                        i += length
        else:
            raise ValueError("This graph does not contain any hyper-edge set, and can't be cast as a flat array.")

    # ------------------------------------------------------------------
    # Padding / un-padding
    # ------------------------------------------------------------------

    def pad(self, target_shape: GraphShape) -> None:
        """Pad hyper-edge sets and address mask to ``target_shape``."""
        if not self.is_single:
            raise ValueError("This graph is not single and cannot be padded.")
        xp = self._backend.xp
        for key, hes_shape in target_shape.hyper_edge_sets.items():
            self.hyper_edge_sets[key].pad(hes_shape)
        self.non_fictitious_addresses = xp.pad(
            self.non_fictitious_addresses, [0, int(target_shape.addresses) - int(self.current_shape.addresses)]
        )
        self.current_shape = target_shape

    def unpad(self) -> None:
        """Remove padding to restore the true shape."""
        for key, hes_shape in self.true_shape.hyper_edge_sets.items():
            self.hyper_edge_sets[key].unpad(hes_shape)
        self.non_fictitious_addresses = self.non_fictitious_addresses[: int(self.true_shape.addresses)]
        self.current_shape = self.true_shape

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def count_connected_components(self) -> tuple[int, any]:
        """
        Count connected components and return per-address component labels.

        :return: ``(num_components, component_labels)``
        :raises ValueError: If the graph is not single.
        """
        backend = self._backend
        xp = backend.xp

        def _max_propagate(*, graph: Graph, h_):
            h_new_ = backend.copy(h_)
            edge_h = {}
            for edge_key, edge in graph.hyper_edge_sets.items():
                edge_h[edge_key] = []
                for _, address_array in edge.port_dict.items():
                    edge_h[edge_key].append(h_new_[address_array.astype(int)])
                edge_h[edge_key] = xp.stack(edge_h[edge_key], axis=0)
                edge_h[edge_key] = xp.max(edge_h[edge_key], axis=0)
                for _, address_array in edge.port_dict.items():
                    new_val = xp.max(
                        xp.stack([edge_h[edge_key], h_new_[address_array.astype(int)]], axis=0),
                        axis=0,
                    )
                    h_new_ = backend.scatter_max(h_new_, address_array.astype(int), new_val)
            return h_new_

        if not self.is_single:
            raise ValueError("Graph is not single.")

        h = xp.arange(len(self.non_fictitious_addresses))
        converged = False
        while not converged:
            h_new = _max_propagate(graph=self, h_=h)
            converged = bool(xp.all(h_new == h))
            h = h_new

        u, indices = xp.unique(h, return_inverse=True)
        return len(u), indices

    def offset_addresses(self, offset) -> None:
        """Add ``offset`` to all port addresses; used before graph concatenation."""
        for e in self.hyper_edge_sets.values():
            e.offset_addresses(offset=offset)

    def quantiles(self, q_list: list[float] | None = None) -> dict:
        """Compute quantiles of all hyper-edge set features.

        :param q_list: Percentiles to compute.
        :return: Dict mapping ``"edge/feature/q%"`` to values.
        """
        if q_list is None:
            q_list = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
        xp = self._backend.xp
        if self.is_single:
            axis = None
        elif self.is_batch:
            axis = 1
        else:
            raise ValueError("This graph is not single or batch and cannot be quantiled.")
        info = {}
        for object_name, hes in self.hyper_edge_sets.items():
            feature_dict = hes.feature_dict
            if feature_dict is not None:
                for feature_name, array in feature_dict.items():
                    if xp.size(array) > 0:
                        values = xp.nanpercentile(array, q=xp.array(q_list), axis=axis)
                        for i, q in enumerate(q_list):
                            info[f"{object_name}/{feature_name}/{q}th-percentile"] = values[i]
        return info


# ---------------------------------------------------------------------------
# Module-level batch/collate functions
# ---------------------------------------------------------------------------


def collate_graphs(graph_list: list[Graph]) -> Graph:
    """
    Collate a list of Graphs into a single batched Graph (all must share ``current_shape``).

    :param graph_list: Non-empty list of Graph instances.
    :returns: Batched Graph.
    :raises ValueError: If ``graph_list`` is empty.
    :raises AssertionError: If ``current_shape`` differs among inputs.
    """
    if not graph_list:
        raise ValueError("collate_graphs requires at least one Graph.")

    cls = type(graph_list[0])
    backend = graph_list[0]._backend
    xp = backend.xp
    first = graph_list[0]

    current_shape = first.current_shape
    for g in graph_list:
        assert g.current_shape == current_shape
    current_shape_batch = collate_shapes([g.current_shape for g in graph_list])
    true_shape_batch = collate_shapes([g.true_shape for g in graph_list])

    hes_batch = {k: collate_hyper_edge_sets([g.hyper_edge_sets[k] for g in graph_list]) for k in first.hyper_edge_sets}

    nfa_batch = (
        xp.stack([g.non_fictitious_addresses for g in graph_list], axis=0)
        if first.non_fictitious_addresses is not None
        else None
    )

    return cls(
        backend=backend,
        hyper_edge_sets=hes_batch,
        non_fictitious_addresses=nfa_batch,
        true_shape=true_shape_batch,
        current_shape=current_shape_batch,
    )


def separate_graphs(graph_batch: Graph) -> list[Graph]:
    """
    Split a batched Graph into a list of single Graphs (reverses :func:`collate_graphs`).

    :param graph_batch: A batched Graph.
    :returns: List of single Graph instances.
    """
    cls = type(graph_batch)
    backend = graph_batch._backend
    xp = backend.xp

    current_shape_list = separate_shapes(graph_batch.current_shape)
    true_shape_list = separate_shapes(graph_batch.true_shape)
    n_batch = len(current_shape_list)

    hes_list_dict = {k: separate_hyper_edge_sets(graph_batch.hyper_edge_sets[k]) for k in graph_batch.hyper_edge_sets}

    nfa_list = (
        xp.unstack(graph_batch.non_fictitious_addresses, axis=0)
        if graph_batch.non_fictitious_addresses is not None
        else [None] * n_batch
    )

    hes_dict_list = [{k: hes_list_dict[k][i] for k in hes_list_dict} for i in range(n_batch)]

    return [
        cls(backend=backend, hyper_edge_sets=e, non_fictitious_addresses=n, true_shape=t, current_shape=c)
        for e, n, t, c in zip(hes_dict_list, nfa_list, true_shape_list, current_shape_list)
    ]


def concatenate_graphs(graph_list: list[Graph]) -> Graph:
    """
    Concatenate multiple single Graphs into one single Graph (no new batch dim).

    :param graph_list: Non-empty list of single Graph instances.
    :return: A new Graph with combined addresses, hyper-edge sets, and shapes.
    :raises ValueError: If ``graph_list`` is empty.
    """
    if not graph_list:
        raise ValueError("graph_list must contain at least one Graph")

    cls = type(graph_list[0])
    backend = graph_list[0]._backend
    xp = backend.xp

    n_addresses_list = [len(g.non_fictitious_addresses) for g in graph_list]
    offset_list = [sum(n_addresses_list[:i]) for i in range(len(n_addresses_list))]

    non_fictitious_addresses = xp.concatenate([g.non_fictitious_addresses for g in graph_list], axis=0)
    true_shape = sum_shapes([g.true_shape for g in graph_list])
    current_shape = sum_shapes([g.current_shape for g in graph_list])

    [g.offset_addresses(offset=offset) for g, offset in zip(graph_list, offset_list)]
    hes = {k: concatenate_hyper_edge_sets([g.hyper_edge_sets[k] for g in graph_list]) for k in graph_list[0].hyper_edge_sets}
    [g.offset_addresses(offset=-offset) for g, offset in zip(graph_list, offset_list)]

    return cls(
        backend=backend,
        hyper_edge_sets=hes,
        non_fictitious_addresses=non_fictitious_addresses,
        true_shape=true_shape,
        current_shape=current_shape,
    )


def union_graphs(graph_list: list[Graph], *, target_shape: GraphShape | None = None, n_graphs: int | None = None) -> Graph:
    """
    Combine single graphs into one disjoint-union graph with segment metadata.

    Unlike :func:`collate_graphs` (which requires identical shapes and adds a batch
    axis), the inputs are concatenated into a single larger graph with offset
    addresses: compute then scales with the *sum* of instance sizes instead of
    ``batch * max_size``. The returned graph carries a ``segments`` entry mapping
    every address and hyper-edge back to its instance, so segment-aware decoders and
    scores can produce per-instance outputs, and :func:`separate_union` can invert
    the operation. Fictitious (padded) entries get the out-of-range id ``n_graphs``
    so scatter-based segment reductions drop them.

    :param graph_list: Non-empty list of single (non-batched, unpadded) graphs.
    :param target_shape: Optional total-size budget to pad the union to. Using a small
        set of budget buckets keeps compiled shapes stable across batches.
    :param n_graphs: Number of instances the union represents; must be
        >= ``len(graph_list)``. Extra instances are empty, allowing a fixed instance
        count across batches. Defaults to ``len(graph_list)``.
    :return: A single Graph with ``segments`` metadata.
    """
    if not graph_list:
        raise ValueError("union_graphs requires at least one Graph.")
    for g in graph_list:
        if not g.is_single:
            raise ValueError("union_graphs requires single (non-batched) graphs.")
    n_real = len(graph_list)
    if n_graphs is None:
        n_graphs = n_real
    if n_graphs < n_real:
        raise ValueError(f"n_graphs ({n_graphs}) must be >= len(graph_list) ({n_real}).")

    backend = graph_list[0]._backend
    xp = backend.xp

    # Per-instance true shapes, padded with empty instances up to n_graphs.
    true_shapes = collate_shapes([g.true_shape for g in graph_list])
    if n_graphs > n_real:
        true_shapes = GraphShape(
            backend=backend,
            hyper_edge_sets={k: xp.pad(v, [(0, n_graphs - n_real)]) for k, v in true_shapes.hyper_edge_sets.items()},
            addresses=xp.pad(true_shapes.addresses, [(0, n_graphs - n_real)]),
        )

    union = concatenate_graphs(graph_list)
    if target_shape is not None:
        union.pad(target_shape)

    def _graph_ids(counts: list[int], total: int) -> np.ndarray:
        ids = np.full(total, n_graphs, dtype=np.int32)
        offset = 0
        for i, count in enumerate(counts):
            ids[offset : offset + count] = i
            offset += count
        return ids

    address_counts = [len(g.non_fictitious_addresses) for g in graph_list]
    address_ids = _graph_ids(address_counts, len(union.non_fictitious_addresses))
    hyper_edge_ids = {
        key: _graph_ids([g.hyper_edge_sets[key].n_obj for g in graph_list], union.hyper_edge_sets[key].n_obj)
        for key in union.hyper_edge_sets
    }

    union.segments = {
        ADDRESS_GRAPH_IDS: xp.array(address_ids),
        HYPER_EDGE_GRAPH_IDS: {k: xp.array(v) for k, v in hyper_edge_ids.items()},
        TRUE_SHAPES: true_shapes,
    }
    return union


def separate_union(graph: Graph) -> list[Graph]:
    """
    Split a disjoint-union graph back into its instances (reverses :func:`union_graphs`).

    Slices every hyper-edge set and the address mask by the per-instance true shapes
    recorded in the segment metadata, restoring local addresses. Padded (fictitious)
    entries and padded empty instances are dropped.

    :param graph: A Graph produced by :func:`union_graphs` (or derived from one, e.g.
        a decoded output that carries the same segment metadata).
    :return: List of single Graph instances, in the original order.
    """
    if graph.segments is None:
        raise ValueError("Graph carries no union segment metadata, impossible to separate.")
    backend = graph._backend
    xp = backend.xp

    shape_list = separate_shapes(graph.segments[TRUE_SHAPES])
    has_addresses = len(graph.non_fictitious_addresses) > 0

    class_counts = {k: [int(s.hyper_edge_sets[k]) for s in shape_list] for k in graph.hyper_edge_sets}
    class_offsets = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in class_counts.items()}
    address_counts = [int(s.addresses) for s in shape_list]
    address_offsets = np.concatenate([[0], np.cumsum(address_counts)])

    graphs = []
    for i, shape in enumerate(shape_list):
        hes_dict = {}
        for key, hes in graph.hyper_edge_sets.items():
            start, count = int(class_offsets[key][i]), class_counts[key][i]
            port_dict = (
                {k: v[start : start + count] - xp.asarray(address_offsets[i], dtype=v.dtype) for k, v in hes.port_dict.items()}
                if hes.port_dict is not None
                else None
            )
            hes_dict[key] = HyperEdgeSet(
                backend=backend,
                port_dict=port_dict,
                feature_array=hes.feature_array[start : start + count] if hes.feature_array is not None else None,
                feature_names=hes.feature_names,
                non_fictitious=hes.non_fictitious[start : start + count],
            )
        if has_addresses:
            nfa = graph.non_fictitious_addresses[int(address_offsets[i]) : int(address_offsets[i]) + address_counts[i]]
            addresses = shape.addresses
        else:
            nfa = xp.zeros([0])
            addresses = xp.array(0)
        true_shape = GraphShape(
            backend=backend,
            hyper_edge_sets={k: shape.hyper_edge_sets[k] for k in graph.hyper_edge_sets},
            addresses=addresses,
        )
        graphs.append(
            type(graph)(
                backend=backend,
                hyper_edge_sets=hes_dict,
                true_shape=true_shape,
                current_shape=true_shape,
                non_fictitious_addresses=nfa,
            )
        )
    return graphs


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_hyper_edge_set_dict_type(hyper_edge_set_dict: dict[str, HyperEdgeSet]) -> None:
    """
    Validate that the mapping is a dict of HyperEdgeSet instances.

    :raises TypeError: If not a dict or if any value is not a HyperEdgeSet.
    """
    if not isinstance(hyper_edge_set_dict, dict):
        raise TypeError("Provided 'hyper_edge_set_dict' is not a 'dict', but a {}.".format(type(hyper_edge_set_dict)))
    for key, hes in hyper_edge_set_dict.items():
        if not isinstance(hes, HyperEdgeSet):
            raise TypeError("Item associated with '{}' key is not a 'HyperEdgeSet'.".format(key))


def check_valid_addresses(hyper_edge_set_dict: dict[str, HyperEdgeSet], n_addresses) -> None:
    """
    Ensure all address indices in each HyperEdgeSet are within ``[0, n_addresses)``.

    :raises AssertionError: If any address is out of range.
    """
    for key, hes in hyper_edge_set_dict.items():
        if hes.port_names is not None:
            xp = hes._backend.xp
            assert xp.all(hes.port_array < n_addresses)


def get_statistics(graph: Graph, axis: int | None = None, norm_graph: Graph | None = None) -> dict:
    """
    Extract summary statistics from each feature array in the graph.

    :param graph: Graph with feature-bearing hyper-edge sets.
    :param axis: Axis for reductions; None = global.
    :param norm_graph: Optional reference graph for normalized metrics.
    :return: Dict mapping ``"edge/feature/stat"`` to values.
    """
    backend = graph._backend
    xp = backend.xp

    for key, hes in graph.hyper_edge_sets.items():
        mask = hes.non_fictitious
        if hes.feature_array is not None:
            fictitious = (mask == 0)[..., None]
            graph.hyper_edge_sets[key].feature_array = xp.where(fictitious, float("nan"), hes.feature_array)

    info = {}
    for object_name, hes in graph.hyper_edge_sets.items():
        feature_dict = hes.feature_dict
        if feature_dict is not None:
            norm_feature_dict = norm_graph.hyper_edge_sets[object_name].feature_dict if norm_graph is not None else None
            for feature_name, array in feature_dict.items():
                if array.size == 0:
                    array = xp.array([[0.0]]) if axis == 1 else xp.array([0.0])

                rmse = xp.sqrt(xp.nanmean(array**2, axis=axis))
                info["{}/{}/rmse".format(object_name, feature_name)] = rmse
                mae = xp.nanmean(xp.abs(array), axis=axis)
                info["{}/{}/mae".format(object_name, feature_name)] = mae

                if norm_feature_dict is not None:
                    norm_array = norm_feature_dict[feature_name]
                    norm_array = norm_array - xp.nanmean(norm_array)
                    info["{}/{}/nrmse".format(object_name, feature_name)] = rmse / (
                        xp.sqrt(xp.nanmean(norm_array**2, axis=axis)) + 1e-9
                    )
                    info["{}/{}/nmae".format(object_name, feature_name)] = mae / (
                        xp.nanmean(xp.abs(norm_array), axis=axis) + 1e-9
                    )

                info["{}/{}/mean".format(object_name, feature_name)] = xp.nanmean(array, axis=axis)
                info["{}/{}/std".format(object_name, feature_name)] = xp.nanstd(array, axis=axis)
                info["{}/{}/max".format(object_name, feature_name)] = xp.nanmax(array, axis=axis)
                percentiles = xp.nanpercentile(array, q=xp.array([90.0, 75.0, 50.0, 25.0, 10.0]), axis=axis)
                info["{}/{}/90th".format(object_name, feature_name)] = percentiles[0]
                info["{}/{}/75th".format(object_name, feature_name)] = percentiles[1]
                info["{}/{}/50th".format(object_name, feature_name)] = percentiles[2]
                info["{}/{}/25th".format(object_name, feature_name)] = percentiles[3]
                info["{}/{}/10th".format(object_name, feature_name)] = percentiles[4]
                info["{}/{}/min".format(object_name, feature_name)] = xp.nanmin(array, axis=axis)
    return info
