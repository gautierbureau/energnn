# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Worked example: learning to solve DC power flow linear systems.

This module implements the :mod:`energnn.problem` interface for a simple supervised task used throughout
the documentation and the :doc:`tutorial </tutorial_notebook>`. Each problem instance is a sparse linear
system :math:`B \\theta = P` describing a DC power grid, where:

- :math:`B` is the (susceptance) matrix of the grid,
- :math:`P` is the vector of active power injections at the buses,
- :math:`\\theta` is the vector of bus phase angles -- the quantity to predict.

The **context** graph carries the ``line`` susceptances and the ``bus`` injections; the **decision** graph
carries a ``phase_angle`` per bus. The objective is the mean-squared error to the oracle solution
:math:`\\theta = B^{-1} P`, so the gradient reduces to ``decision - oracle``.
"""

import copy
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from energnn.graph import GraphStructure, HyperEdgeSetStructure
from energnn.graph import Graph, GraphShape, HyperEdgeSet, JaxBackend, collate_graphs
from ..batch import ProblemBatch
from ..loader import ProblemLoader
from ..problem import Problem

LINEAR_SYSTEM_CONTEXT_STRUCTURE = GraphStructure(
    hyper_edge_sets={
        "line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=["susceptance"]),
        "bus": HyperEdgeSetStructure(port_list=["id"], feature_list=["active_power_injection"]),
    }
)
LINEAR_SYSTEM_DECISION_STRUCTURE = GraphStructure(
    hyper_edge_sets={"bus": HyperEdgeSetStructure(port_list=None, feature_list=["phase_angle"])}
)


class LinearSystemProblemBatch(ProblemBatch):
    """Batched :class:`~energnn.problem.ProblemBatch` for the DC power flow example.

    Holds a batch of contexts (line susceptances and bus injections) together with their oracle solutions
    (bus phase angles). Because the objective is the mean-squared error to the oracle, the gradient is
    simply ``decision - oracle``.

    :param context: Batched context graph.
    :param oracle: Batched oracle graph holding the ground-truth phase angles.
    """

    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = copy.deepcopy(oracle)
        # Vérifier opération
        zero_decision.feature_flat_array = 0.0 * zero_decision.feature_flat_array
        self.zero_decision = zero_decision

    @property
    def decision_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = Graph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[list[float], dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array), axis=1)
        return objective.tolist(), {}

    def save(self, *, path: str) -> None:
        pass


class LinearSystemProblem(Problem):
    """Single :class:`~energnn.problem.Problem` instance of the DC power flow example.

    Represents one linear system :math:`B \\theta = P`: the context carries the line susceptances and bus
    injections, the oracle carries the ground-truth phase angles. The score is the mean-squared error to
    the oracle and the gradient is ``decision - oracle``.

    :param context: Context graph of this instance.
    :param oracle: Oracle graph holding the ground-truth phase angles.
    """

    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = copy.deepcopy(oracle)
        zero_decision.feature_flat_array = 0.0 * zero_decision.feature_flat_array
        self.zero_decision = zero_decision

    @property
    def decision_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = Graph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[float, dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array))
        return float(objective), {}

    def save(self, *, path: str) -> None:
        pass


def _generate_sparse_linear_system(n, m):
    """Generates sparse matrix B and vectors P and theta such that B theta = P for a DC network."""
    # Ensure connectivity by building a spanning tree first
    B = np.zeros((n, n))
    nodes = np.arange(n)
    np.random.shuffle(nodes)
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        weight = np.random.rand() + 0.5
        B[u, v] = B[v, u] = -weight

    # Add remaining m - (n-1) edges
    possible_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j] == 0:
                possible_edges.append((i, j))

    if possible_edges and m > n - 1:
        n_extra = min(m - (n - 1), len(possible_edges))
        idx = np.random.choice(len(possible_edges), n_extra, replace=False)
        for i in idx:
            u, v = possible_edges[i]
            weight = np.random.rand() + 0.5
            B[u, v] = B[v, u] = -weight

    # B is Laplacian matrix-like (off-diagonal < 0, diagonal = -sum(off-diagonal))
    # For a DC network: P = B * theta. B is the susceptance matrix.
    # To have a unique solution, we often fix one node's voltage (slack bus) or add some shunt conductance.
    # Here we'll add a small shunt to ensure invertibility if needed,
    # but the usual DC power flow has sum(P) = 0.
    # Let's make it more generic: B theta = P where B is the susceptance matrix.
    for i in range(n):
        B[i, i] = -np.sum(B[i, :]) + 0.1  # 0.1 for shunt conductance to ground to ensure invertibility

    theta = np.random.randn(n)
    P = B @ theta
    return B, P, theta


class LinearSystemProblemGenerator:
    """Draws random sparse DC power flow systems :math:`B \\theta = P`.

    Each instance builds a connected susceptance matrix ``B`` (a random spanning tree plus extra edges),
    a random phase-angle vector ``theta`` (the oracle) and the induced injections ``P = B @ theta``. The
    number of buses is drawn per instance, up to ``n_max``.

    :param seed: Seed for the NumPy RNG.
    :param n_max: Maximum number of buses of a generated instance.
    """

    __test__ = False

    def __init__(self, *, seed: int = 0, n_max: int = 32):

        self.seed = seed
        self.n_max = n_max

        np.random.seed(seed)

    def generate_problem(self) -> LinearSystemProblem:
        """Draw a single :class:`LinearSystemProblem` with a random number of buses."""
        n = np.random.randint(2, self.n_max + 1)
        m = np.random.randint(n - 1, n * (n - 1) // 2 + 1)
        B, P, theta = _generate_sparse_linear_system(n, m)

        # Context
        # Use line for off-diagonal terms
        rows, cols = np.nonzero(np.triu(B, k=1))
        jax_backend = JaxBackend()
        line = HyperEdgeSet.from_dict(
            backend=jax_backend, port_dict={"from": rows, "to": cols}, feature_dict={"susceptance": -B[rows, cols]}
        )
        bus_context = HyperEdgeSet.from_dict(
            backend=jax_backend, port_dict={"id": np.arange(n)}, feature_dict={"active_power_injection": P}
        )
        context = Graph.from_dict(
            backend=jax_backend, hyper_edge_set_dict={"line": line, "bus": bus_context}, n_addresses=jnp.array(n)
        )

        # Oracle
        # Use bus for the solution (phase angles)
        bus_oracle = HyperEdgeSet.from_dict(backend=jax_backend, port_dict=None, feature_dict={"phase_angle": theta})
        oracle = Graph.from_dict(backend=jax_backend, hyper_edge_set_dict={"bus": bus_oracle}, n_addresses=jnp.array(n))

        return LinearSystemProblem(context=context, oracle=oracle)

    def generate_problem_batch(self, batch_size: int = 8) -> LinearSystemProblemBatch:
        """Draw ``batch_size`` instances and collate them into a :class:`LinearSystemProblemBatch`.

        Each instance is padded to ``n_max`` before collation, so the batch has a homogeneous shape.

        :param batch_size: Number of instances in the batch.
        """
        context_list, oracle_list = [], []

        for _ in range(batch_size):
            problem = self.generate_problem()
            context = problem.context
            oracle = problem.oracle
            context_list.append(context)
            oracle_list.append(oracle)

        jax_backend = JaxBackend()
        max_context_shape = GraphShape(
            backend=jax_backend,
            hyper_edge_sets={
                "line": jnp.array(self.n_max * (self.n_max - 1) // 2),
                "bus": jnp.array(self.n_max),
            },
            addresses=jnp.array(self.n_max),
        )
        max_oracle_shape = GraphShape(
            backend=jax_backend, hyper_edge_sets={"bus": jnp.array(self.n_max)}, addresses=jnp.array(self.n_max)
        )

        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]
        context_batch = collate_graphs(context_list)
        oracle_batch = collate_graphs(oracle_list)

        return LinearSystemProblemBatch(context=context_batch, oracle=oracle_batch)


class LinearSystemProblemLoader(ProblemLoader):
    """:class:`~energnn.problem.ProblemLoader` yielding batches of the DC power flow example.

    Freshly generates each :class:`LinearSystemProblemBatch` on iteration, so it can be passed directly as
    the ``train_loader`` (or ``val_loader``) of a :class:`~energnn.trainer.Trainer`. Reset the RNG by
    re-iterating; use a different ``seed`` for a disjoint dataset (e.g. train vs. validation).

    :param seed: Seed for the NumPy RNG.
    :param dataset_size: Number of problem instances per epoch.
    :param batch_size: Number of instances per batch.
    :param n_max: Maximum number of buses of a generated instance.
    :param shuffle: Kept for interface compatibility (instances are generated on the fly).
    """

    __test__ = False

    def __init__(
        self,
        seed: int = 0,
        dataset_size: int = 32,
        batch_size: int = 8,
        n_max: int = 4,
        shuffle: bool = False,
    ):
        self.seed = seed
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.n_max = n_max
        self.shuffle = shuffle
        self.len = dataset_size
        self.current_step = 0

        self.generator = LinearSystemProblemGenerator(seed=seed, n_max=n_max)

    @property
    def decision_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_CONTEXT_STRUCTURE

    def __iter__(self):
        self.current_step = 0
        np.random.seed(self.seed)
        return self

    def __next__(self) -> LinearSystemProblemBatch:
        if self.current_step >= self.len:
            raise StopIteration
        batch_start = self.current_step
        batch_end = min(self.current_step + self.batch_size, self.len)
        self.current_step = batch_end
        n_batch = batch_end - batch_start
        batch = self.generator.generate_problem_batch(batch_size=n_batch)
        return batch

    def __len__(self):
        return max(self.dataset_size // self.batch_size, 1)
