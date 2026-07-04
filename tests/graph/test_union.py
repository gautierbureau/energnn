# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph import (
    ADDRESS_GRAPH_IDS,
    HYPER_EDGE_GRAPH_IDS,
    TRUE_SHAPES,
    Graph,
    GraphShape,
    HyperEdgeSet,
    separate_union,
    union_graphs,
)


def _make_graph(n_nodes: int, seed: int) -> Graph:
    rng = np.random.default_rng(seed)
    edge = HyperEdgeSet.from_dict(
        port_dict={"src": np.arange(n_nodes), "dst": np.roll(np.arange(n_nodes), 1)},
        feature_dict={"w": rng.normal(size=n_nodes)},
    )
    node = HyperEdgeSet.from_dict(port_dict={"id": np.arange(n_nodes)}, feature_dict={"p": rng.normal(size=n_nodes)})
    return Graph.from_dict(hyper_edge_set_dict={"edge": edge, "node": node}, n_addresses=n_nodes)


def test_union_graphs_segments_and_offsets():
    graphs = [_make_graph(3, 0), _make_graph(5, 1), _make_graph(2, 2)]
    union = union_graphs(graphs)

    assert union.is_single
    assert union.n_union_graphs == 3
    assert union.hyper_edge_sets["node"].n_obj == 10

    segments = union.segments
    np.testing.assert_array_equal(np.array(segments[ADDRESS_GRAPH_IDS]), [0] * 3 + [1] * 5 + [2] * 2)
    np.testing.assert_array_equal(np.array(segments[HYPER_EDGE_GRAPH_IDS]["edge"]), [0] * 3 + [1] * 5 + [2] * 2)
    np.testing.assert_array_equal(np.array(segments[TRUE_SHAPES].addresses), [3, 5, 2])

    # Addresses of the second instance are offset by the first instance's size.
    np.testing.assert_array_equal(
        np.array(union.hyper_edge_sets["node"].port_dict["id"]),
        np.concatenate([np.arange(3), 3 + np.arange(5), 8 + np.arange(2)]),
    )
    # Inputs are restored (concatenation must not permanently offset them).
    np.testing.assert_array_equal(np.array(graphs[1].hyper_edge_sets["node"].port_dict["id"]), np.arange(5))


def test_union_graphs_padding_and_empty_instances():
    graphs = [_make_graph(3, 0), _make_graph(5, 1)]
    target = GraphShape(hyper_edge_sets={"edge": np.array(16), "node": np.array(16)}, addresses=np.array(16))
    union = union_graphs(graphs, target_shape=target, n_graphs=4)

    assert union.n_union_graphs == 4
    assert union.hyper_edge_sets["node"].n_obj == 16
    ids = np.array(union.segments[ADDRESS_GRAPH_IDS])
    # Fictitious (padded) entries carry the out-of-range id n_graphs, so scatter drops them.
    np.testing.assert_array_equal(ids, [0] * 3 + [1] * 5 + [4] * 8)
    np.testing.assert_array_equal(np.array(union.segments[TRUE_SHAPES].addresses), [3, 5, 0, 0])


def test_separate_union_round_trip():
    graphs = [_make_graph(3, 0), _make_graph(5, 1), _make_graph(2, 2)]
    target = GraphShape(hyper_edge_sets={"edge": np.array(16), "node": np.array(16)}, addresses=np.array(16))
    union = union_graphs(graphs, target_shape=target, n_graphs=4)
    separated = separate_union(union)

    assert len(separated) == 4
    assert separated[3].hyper_edge_sets["node"].n_obj == 0
    for original, restored in zip(graphs, separated):
        for key in original.hyper_edge_sets:
            np.testing.assert_array_equal(
                np.array(original.hyper_edge_sets[key].feature_array),
                np.array(restored.hyper_edge_sets[key].feature_array),
            )
            for port in original.hyper_edge_sets[key].port_dict:
                np.testing.assert_array_equal(
                    np.array(original.hyper_edge_sets[key].port_dict[port]),
                    np.array(restored.hyper_edge_sets[key].port_dict[port]),
                )
        np.testing.assert_array_equal(np.array(original.non_fictitious_addresses), np.array(restored.non_fictitious_addresses))


def test_union_graphs_rejects_batched_and_undersized():
    graphs = [_make_graph(3, 0), _make_graph(5, 1)]
    with pytest.raises(ValueError):
        union_graphs([])
    with pytest.raises(ValueError):
        union_graphs(graphs, n_graphs=1)


def test_to_backend_converts_segments():
    from energnn.graph import JaxBackend

    union = union_graphs([_make_graph(3, 0), _make_graph(4, 1)])
    jax_union = union.to_backend(JaxBackend())
    assert jax_union.segments is not None
    np.testing.assert_array_equal(np.array(jax_union.segments[ADDRESS_GRAPH_IDS]), np.array(union.segments[ADDRESS_GRAPH_IDS]))
    assert str(np.array(jax_union.segments[ADDRESS_GRAPH_IDS]).dtype) == "int32"
