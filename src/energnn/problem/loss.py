# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
r"""
Ready-to-use losses for supervised problems.

A supervised problem compares a decision graph :math:`\hat{y}` produced by the GNN with a target
graph :math:`y`. Losses here are written in pure ``jax.numpy`` so that JAX can differentiate them:
the user only declares *which* loss applies to *which* hyper-edge set, and the gradient
:math:`\nabla_{\hat{y}} \mathcal{L}` is obtained with :func:`jax.grad` instead of being hand-written.

Two levels:

- An **object loss** (:class:`Loss` subclasses: :class:`MeanSquaredError`, :class:`SoftmaxCrossEntropy`, ...)
  maps the prediction / target feature arrays of one hyper-edge set, of shapes ``(..., n_obj, n_pred)``
  and ``(..., n_obj, n_target)``, to one value per object, of shape ``(..., n_obj)``.
- A :class:`SupervisedLoss` combines weighted object losses over several hyper-edge sets and
  reduces them to one value per graph: the mean over the *real* objects (padded, fictitious
  objects are masked out) of each hyper-edge set, weighted-summed over hyper-edge sets.

All classes are frozen dataclasses, hence hashable and comparable by value: they are passed as
*static* arguments to :func:`jax.jit`, so a training run compiles the loss (and its gradient)
once per input shape, and two equal loss objects share the same compiled function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp

from energnn.graph import Graph

__all__ = [
    "Loss",
    "MeanSquaredError",
    "MeanAbsoluteError",
    "Huber",
    "BinaryCrossEntropy",
    "SoftmaxCrossEntropy",
    "SigmoidFocalLoss",
    "Accuracy",
    "LossTerm",
    "SupervisedLoss",
]


@dataclass(frozen=True)
class Loss:
    """Base class of object-wise losses.

    Subclasses implement :meth:`__call__`, mapping ``prediction`` of shape ``(..., n_obj, n_pred)``
    and ``target`` of shape ``(..., n_obj, n_target)`` to a per-object value of shape ``(..., n_obj)``.
    Implementations must be pure ``jax.numpy`` code (no Python control flow on array values).
    """

    def __call__(self, prediction: jax.Array, target: jax.Array) -> jax.Array:
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanSquaredError(Loss):
    r"""Squared error averaged over features: :math:`\frac{1}{F}\sum_f (\hat{y}_f - y_f)^2`."""

    def __call__(self, prediction, target):
        return jnp.mean(jnp.square(prediction - target), axis=-1)


@dataclass(frozen=True)
class MeanAbsoluteError(Loss):
    r"""Absolute error averaged over features: :math:`\frac{1}{F}\sum_f |\hat{y}_f - y_f|`."""

    def __call__(self, prediction, target):
        return jnp.mean(jnp.abs(prediction - target), axis=-1)


@dataclass(frozen=True)
class Huber(Loss):
    r"""Huber loss averaged over features, quadratic below ``delta`` and linear above.

    :param delta: Transition point between the quadratic and linear regimes.
    """

    delta: float = 1.0

    def __call__(self, prediction, target):
        abs_err = jnp.abs(prediction - target)
        quadratic = jnp.minimum(abs_err, self.delta)
        linear = abs_err - quadratic
        return jnp.mean(0.5 * jnp.square(quadratic) + self.delta * linear, axis=-1)


# ----------------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------------


def _log_sigmoid_pair(logits):
    """Return ``(log sigmoid(x), log(1 - sigmoid(x)))`` computed stably."""
    return jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)


@dataclass(frozen=True)
class BinaryCrossEntropy(Loss):
    r"""Binary (multi-label) cross-entropy averaged over features.

    Each prediction feature is an independent binary classifier; ``target`` has the same number of
    features and holds labels in :math:`[0, 1]`.

    :param from_logits: If True (default, numerically stable), ``prediction`` holds logits;
        otherwise it holds probabilities (e.g. a decoder with a sigmoid final activation).
    :param label_smoothing: Labels become :math:`y (1 - \epsilon) + \epsilon / 2`.
    :param pos_weight: Weight of the positive term, to counter class imbalance.
    """

    from_logits: bool = True
    label_smoothing: float = 0.0
    pos_weight: float = 1.0

    def __call__(self, prediction, target):
        target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        if self.from_logits:
            log_p, log_not_p = _log_sigmoid_pair(prediction)
        else:
            eps = jnp.finfo(prediction.dtype).eps
            p = jnp.clip(prediction, eps, 1.0 - eps)
            log_p, log_not_p = jnp.log(p), jnp.log1p(-p)
        bce = -(self.pos_weight * target * log_p + (1.0 - target) * log_not_p)
        return jnp.mean(bce, axis=-1)


@dataclass(frozen=True)
class SoftmaxCrossEntropy(Loss):
    r"""Multi-class cross-entropy, one categorical distribution per object.

    ``prediction`` holds ``C`` class scores per object, in the order of the decision's features
    (i.e. the ``feature_list`` of the decoder's output structure). ``target`` holds either

    - **one** feature: the integer class index in ``[0, C)`` (stored as a float), or
    - **C** features: a one-hot / probability vector over the classes.

    :param from_logits: If True (default), ``prediction`` holds logits; otherwise probabilities
        (e.g. a decoder with a softmax final activation).
    :param label_smoothing: Target distribution becomes :math:`y (1 - \epsilon) + \epsilon / C`.
    """

    from_logits: bool = True
    label_smoothing: float = 0.0

    def __call__(self, prediction, target):
        n_classes = prediction.shape[-1]
        if target.shape[-1] == 1 and n_classes > 1:
            target = jax.nn.one_hot(target[..., 0].astype(jnp.int32), n_classes, dtype=prediction.dtype)
        elif target.shape[-1] != n_classes:
            raise ValueError(
                f"SoftmaxCrossEntropy: target has {target.shape[-1]} features, expected 1 (class index) "
                f"or {n_classes} (one-hot / probabilities)."
            )
        target = target * (1.0 - self.label_smoothing) + self.label_smoothing / n_classes
        if self.from_logits:
            log_p = jax.nn.log_softmax(prediction, axis=-1)
        else:
            log_p = jnp.log(jnp.clip(prediction, jnp.finfo(prediction.dtype).eps, 1.0))
        return -jnp.sum(target * log_p, axis=-1)


@dataclass(frozen=True)
class SigmoidFocalLoss(Loss):
    r"""Binary focal loss (Lin et al., 2017) averaged over features, for imbalanced classification.

    :math:`-\alpha_t (1 - p_t)^\gamma \log p_t`, with ``prediction`` holding logits.

    :param gamma: Focusing parameter; ``gamma = 0`` recovers (weighted) binary cross-entropy.
    :param alpha: Weight of the positive class in :math:`[0, 1]`, or ``None`` for no weighting.
    """

    gamma: float = 2.0
    alpha: float | None = 0.25

    def __call__(self, prediction, target):
        log_p, log_not_p = _log_sigmoid_pair(prediction)
        ce = -(target * log_p + (1.0 - target) * log_not_p)
        p = jnp.exp(log_p)
        p_t = target * p + (1.0 - target) * (1.0 - p)
        loss = ce * jnp.power(1.0 - p_t, self.gamma)
        if self.alpha is not None:
            loss = (self.alpha * target + (1.0 - self.alpha) * (1.0 - target)) * loss
        return jnp.mean(loss, axis=-1)


@dataclass(frozen=True)
class Accuracy(Loss):
    """Classification accuracy per object, in ``{0, 1}``. A **metric**: not differentiable.

    Meant as the ``metric`` of a supervised problem (scores maximized: ``optim_mode="maximize"``).

    - Multi-class (``n_pred > 1``): ``argmax`` of the prediction compared with the target class,
      the target being a class index (one feature) or a one-hot vector (``argmax`` taken).
    - Binary (``n_pred == 1``): prediction thresholded at ``threshold``, then averaged over features.

    :param threshold: Decision threshold for binary predictions (0.0 for logits, 0.5 for probabilities).
    """

    threshold: float = 0.0

    def __call__(self, prediction, target):
        if prediction.shape[-1] > 1:
            predicted = jnp.argmax(prediction, axis=-1)
            if target.shape[-1] == 1:
                expected = target[..., 0].astype(jnp.int32)
            else:
                expected = jnp.argmax(target, axis=-1)
            return (predicted == expected).astype(prediction.dtype)
        predicted = (prediction > self.threshold).astype(prediction.dtype)
        return jnp.mean((predicted == jnp.round(target)).astype(prediction.dtype), axis=-1)


# ----------------------------------------------------------------------------------------------
# Graph-level aggregation
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LossTerm:
    """One object loss applied to one hyper-edge set of the decision / target graphs.

    :param hyper_edge_set: Name of the hyper-edge set (present in both decision and target graphs).
    :param loss: Object loss applied to its feature arrays.
    :param weight: Weight of this term in the graph-level sum.
    """

    hyper_edge_set: str
    loss: Loss
    weight: float = 1.0


@dataclass(frozen=True)
class SupervisedLoss:
    r"""Graph-level supervised loss, a weighted sum of per hyper-edge set masked means.

    .. math::
        \mathcal{L}(\hat{y}, y) = \sum_{c} w_c \frac{1}{|\mathcal{E}^c|}
        \sum_{e \in \mathcal{E}^c} \ell_c(\hat{y}_e, y_e),

    where :math:`\mathcal{E}^c` are the *real* (non-fictitious) objects of class :math:`c`, taken
    from the target's ``non_fictitious`` mask so padding never contributes.

    Build it with :meth:`from_dict`, e.g.
    ``SupervisedLoss.from_dict({"bus": SoftmaxCrossEntropy(), "line": (MeanSquaredError(), 0.1)})``.
    """

    terms: tuple[LossTerm, ...]

    def __post_init__(self):
        if len(self.terms) == 0:
            raise ValueError("SupervisedLoss needs at least one LossTerm.")

    @classmethod
    def from_dict(cls, losses: Mapping[str, Loss | tuple[Loss, float]]) -> SupervisedLoss:
        """Build from ``{hyper_edge_set: loss}`` or ``{hyper_edge_set: (loss, weight)}``."""
        terms = []
        for name in sorted(losses):
            spec = losses[name]
            loss, weight = spec if isinstance(spec, tuple) else (spec, 1.0)
            if not isinstance(loss, Loss):
                raise TypeError(f"Loss for hyper-edge set '{name}' must be a Loss, got {type(loss)}.")
            terms.append(LossTerm(hyper_edge_set=name, loss=loss, weight=float(weight)))
        return cls(terms=tuple(terms))

    def per_graph(self, decision: Graph, target: Graph) -> jax.Array:
        """Loss value of each graph: shape ``(batch,)`` for batched graphs, ``()`` for single graphs."""
        total = 0.0
        for term in self.terms:
            pred_hes = decision.hyper_edge_sets[term.hyper_edge_set]
            tgt_hes = target.hyper_edge_sets[term.hyper_edge_set]
            prediction = pred_hes.feature_array
            mask = jnp.asarray(tgt_hes.non_fictitious, dtype=prediction.dtype)
            real = (mask > 0)[..., None]
            # Padded objects are fed zeros (finite for every loss, logits or probabilities) *before* the
            # loss: masking only the output would still let a NaN/inf leak into the gradient (0 * nan).
            prediction = jnp.where(real, prediction, 0.0)
            target_features = jnp.where(real, tgt_hes.feature_array.astype(prediction.dtype), 0.0)
            per_object = term.loss(prediction, target_features)
            n_real = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
            total = total + term.weight * jnp.sum(per_object * mask, axis=-1) / n_real
        return total

    def __call__(self, decision: Graph, target: Graph) -> jax.Array:
        """Scalar training objective: the mean of :meth:`per_graph` over the batch."""
        return jnp.mean(self.per_graph(decision, target))
