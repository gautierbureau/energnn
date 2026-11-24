# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph import Graph, HyperEdgeSet, JaxBackend
from energnn.problem import SupervisedProblem, SupervisedProblemBatch
from energnn.problem.loss import (
    Accuracy,
    BinaryCrossEntropy,
    Huber,
    LossTerm,
    MeanAbsoluteError,
    MeanSquaredError,
    SigmoidFocalLoss,
    SoftmaxCrossEntropy,
    SupervisedLoss,
)
from energnn.problem.supervised import jit_loss_value_and_decision_gradient

B = JaxBackend()


def _graph(features: np.ndarray, mask: np.ndarray, names: list[str]) -> Graph:
    """Batched graph with a single 'bus' hyper-edge set; features (batch, n_obj, n_feat), mask (batch, n_obj)."""
    hes = HyperEdgeSet(
        backend=B,
        port_dict=None,
        feature_array=jnp.asarray(features, dtype=jnp.float32),
        feature_names={n: jnp.array([i]) for i, n in enumerate(names)},
        non_fictitious=jnp.asarray(mask, dtype=jnp.float32),
    )
    return Graph(
        backend=B,
        hyper_edge_sets={"bus": hes},
        true_shape=jnp.array(0),
        current_shape=jnp.array(0),
        non_fictitious_addresses=jnp.ones(features.shape[:2]),
    )


# ---------------------------------------------------------------- object losses vs closed forms


def test_regression_losses():
    p, t = jnp.array([[1.0, -2.0]]), jnp.array([[0.0, 1.0]])
    np.testing.assert_allclose(MeanSquaredError()(p, t), [(1 + 9) / 2])
    np.testing.assert_allclose(MeanAbsoluteError()(p, t), [(1 + 3) / 2])
    # Huber(1): 0.5*1 for |e|=1, 1*(3-0.5)=2.5 for |e|=3
    np.testing.assert_allclose(Huber(delta=1.0)(p, t), [(0.5 + 2.5) / 2])


def test_softmax_cross_entropy_index_and_one_hot_agree():
    logits = jnp.array([[2.0, 0.5, -1.0], [0.0, 0.0, 3.0]])
    expected = -jax.nn.log_softmax(logits)[jnp.arange(2), jnp.array([0, 2])]
    np.testing.assert_allclose(SoftmaxCrossEntropy()(logits, jnp.array([[0.0], [2.0]])), expected, rtol=1e-6)
    np.testing.assert_allclose(SoftmaxCrossEntropy()(logits, jax.nn.one_hot(jnp.array([0, 2]), 3)), expected, rtol=1e-6)
    # probabilities input gives the same value
    np.testing.assert_allclose(
        SoftmaxCrossEntropy(from_logits=False)(jax.nn.softmax(logits), jnp.array([[0.0], [2.0]])), expected, rtol=1e-5
    )
    with pytest.raises(ValueError):
        SoftmaxCrossEntropy()(logits, jnp.zeros((2, 2)))


def test_binary_losses():
    logits, y = jnp.array([[3.0], [-1.0]]), jnp.array([[1.0], [0.0]])
    p = jax.nn.sigmoid(logits)
    bce = -(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))[:, 0]
    np.testing.assert_allclose(BinaryCrossEntropy()(logits, y), bce, rtol=1e-6)
    np.testing.assert_allclose(BinaryCrossEntropy(from_logits=False)(p, y), bce, rtol=1e-5)
    # gamma=0, alpha=None: focal loss is BCE
    np.testing.assert_allclose(SigmoidFocalLoss(gamma=0.0, alpha=None)(logits, y), bce, rtol=1e-6)
    assert jnp.all(SigmoidFocalLoss()(logits, y) < bce)


def test_accuracy():
    np.testing.assert_allclose(Accuracy()(jnp.array([[2.0, 1.0], [0.0, 1.0]]), jnp.array([[0.0], [0.0]])), [1.0, 0.0])
    np.testing.assert_allclose(Accuracy()(jnp.array([[2.0], [-1.0]]), jnp.array([[1.0], [1.0]])), [1.0, 0.0])


def test_losses_are_hashable_by_value():
    assert SupervisedLoss.from_dict({"bus": SoftmaxCrossEntropy()}) == SupervisedLoss.from_dict({"bus": SoftmaxCrossEntropy()})
    assert hash(SupervisedLoss.from_dict({"bus": (Huber(0.5), 2.0)})) == hash(
        SupervisedLoss(terms=(LossTerm("bus", Huber(0.5), 2.0),))
    )
    assert Huber(0.5) != Huber(1.0)


# ---------------------------------------------------------------- graph-level loss, masking, gradient


def test_padding_is_masked_out():
    pred = np.zeros((2, 3, 1))
    pred[:, 2] = np.nan  # padded object, would poison the loss without the mask
    tgt = np.ones((2, 3, 1))
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    loss = SupervisedLoss.from_dict({"bus": MeanSquaredError()})
    per_graph = loss.per_graph(_graph(pred, mask, ["y"]), _graph(tgt, mask, ["y"]))
    np.testing.assert_allclose(per_graph, [1.0, 1.0])
    _, grad = jit_loss_value_and_decision_gradient(loss, _graph(pred, mask, ["y"]), _graph(tgt, mask, ["y"]))
    g = np.asarray(grad.hyper_edge_sets["bus"].feature_array)
    assert np.all(np.isfinite(g))
    assert np.all(g[mask == 0] == 0)


def test_autodiff_gradient_matches_closed_form():
    rng = np.random.default_rng(0)
    pred, tgt = rng.normal(size=(4, 5, 2)), rng.normal(size=(4, 5, 2))
    mask = np.ones((4, 5))
    mask[0, 3:] = 0
    loss = SupervisedLoss.from_dict({"bus": MeanSquaredError()})
    value, grad = jit_loss_value_and_decision_gradient(loss, _graph(pred, mask, ["a", "b"]), _graph(tgt, mask, ["a", "b"]))
    # d/dpred of mean_b [ sum_real mean_f (p-t)^2 / n_real_b ]
    n_real = mask.sum(-1)[:, None, None]
    expected = 2 * (pred - tgt) / 2 / n_real / 4 * mask[..., None]
    np.testing.assert_allclose(np.asarray(grad.hyper_edge_sets["bus"].feature_array), expected, rtol=1e-5, atol=1e-7)
    # other leaves are zero cotangents with the decision's pytree structure
    assert jax.tree.structure(grad) == jax.tree.structure(_graph(pred, mask, ["a", "b"]))
    assert float(jnp.sum(grad.hyper_edge_sets["bus"].non_fictitious)) == 0.0


def test_gradient_compiled_once_across_batches_and_loss_instances():
    jit_loss_value_and_decision_gradient.clear_cache()
    rng = np.random.default_rng(1)
    mask = np.ones((3, 4))
    for _ in range(5):
        batch = SupervisedProblemBatch(
            context=_graph(rng.normal(size=(3, 4, 1)), mask, ["x"]),
            target=_graph(rng.integers(0, 3, size=(3, 4, 1)).astype(float), mask, ["label"]),
            loss={"bus": SoftmaxCrossEntropy(label_smoothing=0.1)},  # a new, equal, loss object each time
        )
        batch.get_gradient(decision=_graph(rng.normal(size=(3, 4, 3)), mask, ["c0", "c1", "c2"]))
    assert jit_loss_value_and_decision_gradient._cache_size() == 1


def test_supervised_problem_score_and_structures():
    mask = np.ones((1, 2))
    target = _graph(np.array([[[1.0], [0.0]]]), mask, ["label"])
    batch = SupervisedProblemBatch(
        context=_graph(np.zeros((1, 2, 1)), mask, ["x"]),
        target=target,
        loss={"bus": SoftmaxCrossEntropy()},
        metric={"bus": Accuracy()},
    )
    decision = _graph(np.array([[[0.0, 5.0], [0.0, 5.0]]]), mask, ["c0", "c1"])
    score, _ = batch.get_score(decision=decision)
    np.testing.assert_allclose(score, [0.5])
    assert batch.decision_structure.hyper_edge_sets["bus"].feature_list == ["label"]
    assert batch.context_structure.hyper_edge_sets["bus"].feature_list == ["x"]
    with pytest.raises(KeyError):
        SupervisedProblemBatch(context=target, target=target, loss={"line": MeanSquaredError()})

    # single (unbatched) instance: same API, scalar score
    unbatch = lambda g: jax.tree.map(lambda x: x[0] if getattr(x, "ndim", 0) > 0 else x, g)  # noqa: E731
    single = SupervisedProblem(context=unbatch(batch.context), target=unbatch(target), loss={"bus": SoftmaxCrossEntropy()})
    value, _ = single.get_score(decision=unbatch(decision))
    expected = SoftmaxCrossEntropy()(
        decision.hyper_edge_sets["bus"].feature_array, target.hyper_edge_sets["bus"].feature_array
    )
    assert value == pytest.approx(float(expected.mean()), rel=1e-6)
