# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Supervised training with a ready-made loss: no hand-written gradient, one compilation per run."""
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from energnn.graph import GraphStructure, HyperEdgeSet, HyperEdgeSetStructure
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from energnn.problem import ProblemLoader, SupervisedProblemBatch
from energnn.problem.example import LinearSystemProblemLoader
from energnn.problem.loss import Accuracy, SoftmaxCrossEntropy
from energnn.trainer import Trainer

# Decoder outputs 2 logits per bus; the target holds the class index (sign of the phase angle).
LOGITS_STRUCTURE = GraphStructure(
    hyper_edge_sets={"bus": HyperEdgeSetStructure(port_list=None, feature_list=["negative", "positive"])}
)


class BusSignClassificationLoader(ProblemLoader):
    """Wraps the linear-system loader into a 2-class bus classification task."""

    def __init__(self, seed: int = 0, dataset_size: int = 16, batch_size: int = 4):
        self.inner = LinearSystemProblemLoader(seed=seed, dataset_size=dataset_size, batch_size=batch_size, n_max=6)

    def __iter__(self):
        self.it = iter(self.inner)
        return self

    def __next__(self) -> SupervisedProblemBatch:
        batch = next(self.it)
        oracle = batch.oracle
        hes = oracle.hyper_edge_sets["bus"]
        label = HyperEdgeSet(
            backend=hes._backend,
            port_dict=None,
            feature_array=(hes.feature_array > 0).astype(jnp.float32),
            feature_names={"label": jnp.zeros_like(next(iter(hes.feature_names.values())))},
            non_fictitious=hes.non_fictitious,
        )
        target = jax.tree.map(lambda x: x, oracle)  # shallow copy
        target.hyper_edge_sets = {"bus": label}
        return SupervisedProblemBatch(
            context=batch.context,
            target=target,
            loss={"bus": SoftmaxCrossEntropy()},
            metric={"bus": Accuracy()},
            context_structure=self.inner.context_structure,
            decision_structure=LOGITS_STRUCTURE,
        )

    def __len__(self):
        return len(self.inner)

    @property
    def context_structure(self):
        return self.inner.context_structure

    @property
    def decision_structure(self):
        return LOGITS_STRUCTURE


def _model(seed=0):
    loader = BusSignClassificationLoader()
    return TinyRecurrentEquivariantGNN(in_structure=loader.context_structure, out_structure=LOGITS_STRUCTURE, seed=seed)


def _params(model):
    return [np.asarray(x) for x in jax.tree.leaves(nnx.state(model, nnx.Param))]


def test_fused_step_matches_generic_vjp_step():
    """Fused (value_and_grad in one jit) and generic (forward / autodiff get_gradient / VJP) give the same update."""
    batch = next(iter(BusSignClassificationLoader()))
    fused, generic = _model(), _model()
    Trainer(model=fused, gradient_transformation=optax.sgd(0.5)).training_step(batch, get_info=False)
    Trainer(model=generic, gradient_transformation=optax.sgd(0.5), fuse_supervised_step=False).training_step(
        batch, get_info=False
    )
    before = _params(_model())
    assert any(not np.allclose(a, b) for a, b in zip(before, _params(fused)))
    for a, b in zip(_params(fused), _params(generic)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_fused_supervised_step_traced_once_over_training():
    """Over a full training run (several epochs, logging on/off), the step is compiled once per get_info value."""
    trace_count = {True: 0, False: 0}
    original = Trainer._supervised_step

    def counting(model, optimizer, context, target, loss, get_info):
        trace_count[get_info] += 1
        return original(model, optimizer, context, target, loss, get_info)

    with mock.patch.object(Trainer, "_supervised_step", staticmethod(counting)):
        trainer = Trainer(model=_model(), gradient_transformation=optax.adam(1e-2))
        trainer.train(train_loader=BusSignClassificationLoader(), n_epochs=3, progress_bar=False, log_period=None)
        for i, batch in enumerate(BusSignClassificationLoader()):
            trainer.training_step(batch, get_info=(i % 2 == 0))
    assert trace_count == {True: 1, False: 1}


def test_supervised_training_improves_loss_and_evaluates_metric():
    loader = BusSignClassificationLoader(dataset_size=16, batch_size=4)
    trainer = Trainer(model=_model(), gradient_transformation=optax.adam(1e-2))
    batch = next(iter(loader))

    def mean_loss():
        decision, _ = trainer.model.forward_batch(graph=batch.context)
        return float(batch.loss(decision, batch.target))

    loss_before = mean_loss()
    best = trainer.train(train_loader=loader, val_loader=loader, n_epochs=10, progress_bar=False, optim_mode="maximize")
    assert mean_loss() < loss_before
    assert 0.0 <= best <= 1.0  # validation score is the accuracy metric
