# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from energnn.model.decoder.invariant_decoder import SumInvariantDecoder
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from energnn.model.utils import MLP
from energnn.problem.example import LinearSystemProblemLoader
from energnn.problem.example.linear_system import LinearSystemProblemGenerator
from energnn.trainer import Trainer


def _model(loader):
    return TinyRecurrentEquivariantGNN(in_structure=loader.context_structure, out_structure=loader.decision_structure)


def test_union_forward_matches_dense_per_instance():
    """The union path must produce the same per-instance outputs as the dense vmapped path."""
    loader = LinearSystemProblemLoader(seed=1, n_max=12)
    model = _model(loader)
    model.eval()

    dense = LinearSystemProblemGenerator(seed=9, n_max=12).generate_problem_batch(batch_size=4, mode="dense")
    union = LinearSystemProblemGenerator(seed=9, n_max=12).generate_problem_batch(batch_size=4, mode="union")

    forward = nnx.jit(lambda m, g: m.forward_batch(graph=g, get_info=False)[0])
    out_dense = forward(model, dense.context)
    out_union = forward(model, union.context)

    dense_features = np.array(out_dense.hyper_edge_sets["bus"].feature_array)
    union_features = np.array(out_union.hyper_edge_sets["bus"].feature_array)
    sizes = [int(s) for s in np.array(union.context.segments["true_shapes"].hyper_edge_sets["bus"])]
    offset = 0
    for i, size in enumerate(sizes):
        np.testing.assert_allclose(union_features[offset : offset + size], dense_features[i, :size], rtol=2e-5, atol=1e-6)
        offset += size


def test_union_scores_match_dense_reference():
    """Union per-instance scores equal the masked per-instance means of the dense outputs."""
    loader = LinearSystemProblemLoader(seed=1, n_max=12)
    model = _model(loader)
    model.eval()

    dense = LinearSystemProblemGenerator(seed=9, n_max=12).generate_problem_batch(batch_size=4, mode="dense")
    union = LinearSystemProblemGenerator(seed=9, n_max=12).generate_problem_batch(batch_size=4, mode="union")

    forward = nnx.jit(lambda m, g: m.forward_batch(graph=g, get_info=False)[0])
    out_dense = forward(model, dense.context)
    out_union = forward(model, union.context)

    union_scores, _ = union.get_score(decision=out_union)
    error = np.square(
        np.array(out_dense.hyper_edge_sets["bus"].feature_array)[..., 0]
        - np.array(dense.oracle.hyper_edge_sets["bus"].feature_array)[..., 0]
    )
    mask = np.array(dense.oracle.hyper_edge_sets["bus"].non_fictitious)
    reference = [(error[i] * mask[i]).sum() / mask[i].sum() for i in range(4)]
    np.testing.assert_allclose(union_scores, reference, rtol=2e-5)


def test_invariant_decoder_segment_sums():
    """On a union graph, invariant decoders return one output per instance."""
    union = LinearSystemProblemGenerator(seed=3, n_max=10).generate_problem_batch(batch_size=3, mode="union").context
    psi = MLP(in_size=4, hidden_sizes=[], out_size=4, seed=0)
    phi = MLP(in_size=4, hidden_sizes=[], out_size=2, seed=1)
    decoder = SumInvariantDecoder(psi=psi, phi=phi)

    coordinates = jnp.array(
        np.random.default_rng(0).normal(size=(union.non_fictitious_addresses.shape[0], 4)), dtype=jnp.float32
    )
    out, _ = decoder(graph=union, coordinates=coordinates)
    assert out.shape == (3, 2)

    address_sizes = [int(s) for s in np.array(union.segments["true_shapes"].addresses)]
    offset = 0
    for i, size in enumerate(address_sizes):
        reference = phi(jnp.sum(psi(coordinates[offset : offset + size]), axis=0))
        np.testing.assert_allclose(np.array(out[i]), np.array(reference), rtol=2e-5, atol=1e-6)
        offset += size


def test_union_training_converges():
    """End-to-end training in union mode goes through the unchanged Trainer and converges."""
    train_loader = LinearSystemProblemLoader(seed=1, dataset_size=32, batch_size=8, n_max=8, mode="union")
    val_loader = LinearSystemProblemLoader(seed=2, dataset_size=16, batch_size=8, n_max=8, mode="union")
    model = _model(train_loader)
    trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-2))

    score_before, _ = trainer.eval(val_loader)
    trainer.train(train_loader=train_loader, n_epochs=5, progress_bar=False)
    score_after, _ = trainer.eval(val_loader)
    assert score_after < score_before
