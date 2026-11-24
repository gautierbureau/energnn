# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
r"""
Supervised problems defined by a target graph and a loss only.

For a supervised task (regression, node / edge classification, ...), the objective is a loss
:math:`\mathcal{L}(\hat{y}, y)` between the decision :math:`\hat{y}` and a known target :math:`y`.
Instead of hand-writing :meth:`~energnn.problem.ProblemBatch.get_gradient`, the user gives a
:class:`~energnn.problem.loss.SupervisedLoss` (or a ``{hyper_edge_set: Loss}`` dict) and the
gradient :math:`\nabla_{\hat{y}} \mathcal{L}` is computed by :func:`jax.grad`.

JIT: the gradient and score functions are module-level :func:`jax.jit` functions taking the loss as
a *static* argument. They compile once per (loss value, graph shapes) and are reused across
batches, problem instances and trainers. Keep batch shapes fixed (pad every batch to the same
:class:`~energnn.graph.GraphShape`, as the loaders do) to stay on a single compilation.

The :class:`~energnn.trainer.Trainer` recognizes :class:`SupervisedProblemBatch` and runs the whole
step (forward, loss, backward, optimizer update) as one compiled function instead.
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from energnn.graph import Graph, GraphStructure, HyperEdgeSet, HyperEdgeSetStructure
from .batch import ProblemBatch
from .loss import Loss, SupervisedLoss
from .problem import Problem

LossSpec = SupervisedLoss | Mapping[str, Loss | tuple[Loss, float]]


def as_supervised_loss(loss: LossSpec) -> SupervisedLoss:
    """Normalize a loss specification (a :class:`SupervisedLoss` or a dict) into a :class:`SupervisedLoss`."""
    if isinstance(loss, SupervisedLoss):
        return loss
    if isinstance(loss, Mapping):
        return SupervisedLoss.from_dict(loss)
    raise TypeError(f"Expected a SupervisedLoss or a {{hyper_edge_set: Loss}} dict, got {type(loss)}.")


def replace_feature_arrays(graph: Graph, feature_arrays: Mapping[str, jax.Array]) -> Graph:
    """Shallow copy of ``graph`` whose hyper-edge sets listed in ``feature_arrays`` get new feature arrays."""
    hyper_edge_sets = {
        k: (
            HyperEdgeSet(
                backend=hes._backend,
                port_dict=hes.port_dict,
                feature_array=feature_arrays[k],
                feature_names=hes.feature_names,
                non_fictitious=hes.non_fictitious,
            )
            if k in feature_arrays
            else hes
        )
        for k, hes in graph.hyper_edge_sets.items()
    }
    return Graph(
        backend=graph._backend,
        hyper_edge_sets=hyper_edge_sets,
        true_shape=graph.true_shape,
        current_shape=graph.current_shape,
        non_fictitious_addresses=graph.non_fictitious_addresses,
    )


def _feature_arrays(graph: Graph) -> dict[str, jax.Array]:
    return {k: hes.feature_array for k, hes in graph.hyper_edge_sets.items() if hes.feature_array is not None}


def loss_value_and_decision_gradient(loss: SupervisedLoss, decision: Graph, target: Graph) -> tuple[jax.Array, Graph]:
    r"""Mean loss over the batch and its gradient :math:`\nabla_{\hat{y}} \mathcal{L}` as a Graph.

    Only feature arrays are differentiated; every other leaf of the returned gradient graph
    (masks, shapes, ...) is zero, so it can be used directly as the VJP cotangent of the decision.
    Not jitted itself: see :func:`jit_loss_value_and_decision_gradient`.
    """

    def objective(features):
        return loss(replace_feature_arrays(decision, features), target)

    value, feature_grads = jax.value_and_grad(objective)(_feature_arrays(decision))
    gradient = jax.tree.map(jnp.zeros_like, decision)
    return value, replace_feature_arrays(gradient, feature_grads)


# Compiled once per (loss, shapes): the loss is hashable by value, hence a valid static argument.
jit_loss_value_and_decision_gradient = jax.jit(loss_value_and_decision_gradient, static_argnums=0)
jit_per_graph = jax.jit(lambda loss, decision, target: loss.per_graph(decision, target), static_argnums=0)


def _structure_of(graph: Graph) -> GraphStructure:
    """Infer the :class:`GraphStructure` (ports and ordered feature names) of a graph."""
    hyper_edge_sets = {}
    for k, hes in graph.hyper_edge_sets.items():
        port_list = sorted(hes.port_dict) if hes.port_dict is not None else None
        feature_list = None
        if hes.feature_names is not None:
            feature_list = sorted(hes.feature_names, key=lambda n: int(np.asarray(hes.feature_names[n]).ravel()[0]))
        hyper_edge_sets[k] = HyperEdgeSetStructure(port_list=port_list, feature_list=feature_list)
    return GraphStructure(hyper_edge_sets=hyper_edge_sets)


class _SupervisedMixin:
    """Shared implementation of single and batched supervised problems."""

    def _init_supervised(self, *, context, target, loss, metric, context_structure, decision_structure):
        self.context = context
        self.target = target
        self.loss = as_supervised_loss(loss)
        self.metric = as_supervised_loss(metric) if metric is not None else self.loss
        for term in self.loss.terms:
            if term.hyper_edge_set not in target.hyper_edge_sets:
                raise KeyError(f"Loss term on '{term.hyper_edge_set}' but the target has no such hyper-edge set.")
        self._context_structure = context_structure
        self._decision_structure = decision_structure

    @property
    def context_structure(self) -> GraphStructure:
        if self._context_structure is None:
            self._context_structure = _structure_of(self.context)
        return self._context_structure

    @property
    def decision_structure(self) -> GraphStructure:
        if self._decision_structure is None:
            self._decision_structure = _structure_of(self.target)
        return self._decision_structure

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context graph :math:`x`."""
        return self.context, {}

    def get_target(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the target graph :math:`y` (same structure and padding as the decision)."""
        return self.target, {}

    def get_gradient(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        r"""Gradient :math:`\nabla_{\hat{y}} \mathcal{L}` of the batch-mean loss, by automatic differentiation."""
        target, _ = self.get_target(step=step)
        value, gradient = jit_loss_value_and_decision_gradient(self.loss, decision, target)
        return gradient, ({"loss": value} if get_info else {})

    def save(self, *, path: str) -> None:
        self.context.to_pickle(f"{path}_context.pkl")
        self.target.to_pickle(f"{path}_target.pkl")


class SupervisedProblemBatch(_SupervisedMixin, ProblemBatch):
    r"""A batch of supervised problems: a context batch, a target batch and a loss.

    No gradient to write: :meth:`get_gradient` differentiates the loss with JAX, and the
    :class:`~energnn.trainer.Trainer` fuses forward, loss, backward and update into one compiled step.

    Example (bus classification into 3 classes, decoder outputs 3 logits per bus)::

        batch = SupervisedProblemBatch(
            context=context_batch,
            target=label_batch,  # "bus" hyper-edge set with one feature: the class index
            loss={"bus": SoftmaxCrossEntropy()},
            metric={"bus": Accuracy()},  # train with optim_mode="maximize"
        )

    :param context: Batched context graph :math:`x`.
    :param target: Batched target graph :math:`y`, padded like the decision (same hyper-edge sets and
        object counts); its ``non_fictitious`` masks select the objects the loss is averaged over.
    :param loss: Differentiable training loss: a :class:`SupervisedLoss` or ``{hyper_edge_set: Loss}``.
    :param metric: Score reported by :meth:`get_score` (validation / checkpointing); defaults to ``loss``.
    :param context_structure: Optional explicit context structure (inferred from ``context`` otherwise).
    :param decision_structure: Optional explicit decision structure (inferred from ``target`` otherwise).
    """

    def __init__(
        self,
        *,
        context: Graph,
        target: Graph,
        loss: LossSpec,
        metric: LossSpec | None = None,
        context_structure: GraphStructure | None = None,
        decision_structure: GraphStructure | None = None,
    ):
        self._init_supervised(
            context=context,
            target=target,
            loss=loss,
            metric=metric,
            context_structure=context_structure,
            decision_structure=decision_structure,
        )

    def get_score(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[list[float], dict]:
        """Metric (by default the loss) of each graph of the batch."""
        target, _ = self.get_target(step=step)
        return np.asarray(jit_per_graph(self.metric, decision, target)).tolist(), {}


class SupervisedProblem(_SupervisedMixin, Problem):
    """A single supervised problem instance; see :class:`SupervisedProblemBatch`."""

    def __init__(
        self,
        *,
        context: Graph,
        target: Graph,
        loss: LossSpec,
        metric: LossSpec | None = None,
        context_structure: GraphStructure | None = None,
        decision_structure: GraphStructure | None = None,
    ):
        self._init_supervised(
            context=context,
            target=target,
            loss=loss,
            metric=metric,
            context_structure=context_structure,
            decision_structure=decision_structure,
        )

    def get_score(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[float, dict]:
        """Metric (by default the loss) of the decision."""
        target, _ = self.get_target(step=step)
        return float(jit_per_graph(self.metric, decision, target)), {}
