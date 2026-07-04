# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from energnn.graph import GraphStructure, HyperEdgeSetStructure
from energnn.graph import Graph, GraphShape, HyperEdgeSet, collate_graphs
from ..batch import ProblemBatch
from ..loader import ProblemLoader
from ..problem import Problem


def _tree_copy(graph: Graph) -> Graph:
    """
    Structural copy of a JAX-backed graph.

    Rebuilds all the containers (so in-place structural mutations of the copy do not
    affect the original) while sharing the immutable JAX array leaves. This is much
    cheaper than ``copy.deepcopy``, which would also copy every array buffer.
    """
    return jax.tree.map(lambda x: x, graph)


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
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = _tree_copy(oracle)
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
        return _tree_copy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return _tree_copy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return _tree_copy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        gradient = _tree_copy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[list[float], dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        gradient = _tree_copy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array), axis=1)
        return objective.tolist(), {}

    def save(self, *, path: str) -> None:
        pass


class LinearSystemProblem(Problem):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = _tree_copy(oracle)
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
        return _tree_copy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return _tree_copy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return _tree_copy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        gradient = _tree_copy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[float, dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        gradient = _tree_copy(decision)
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
    __test__ = False
    """Generates random sparse linear systems."""

    def __init__(self, *, seed: int = 0, n_max: int = 32):

        self.seed = seed
        self.n_max = n_max

        np.random.seed(seed)

    def _generate_numpy_problem(self) -> tuple[Graph, Graph]:
        """Generate a (context, oracle) pair on the NumPy backend.

        Construction, validation, and padding of many small, randomly-sized graphs are
        much cheaper on NumPy: doing them on the JAX backend triggers one XLA
        compilation per new array shape.
        """
        n = np.random.randint(2, self.n_max + 1)
        m = np.random.randint(n - 1, n * (n - 1) // 2 + 1)
        B, P, theta = _generate_sparse_linear_system(n, m)

        # Context
        # Use line for off-diagonal terms
        rows, cols = np.nonzero(np.triu(B, k=1))
        line = HyperEdgeSet.from_dict(port_dict={"from": rows, "to": cols}, feature_dict={"susceptance": -B[rows, cols]})
        bus_context = HyperEdgeSet.from_dict(port_dict={"id": np.arange(n)}, feature_dict={"active_power_injection": P})
        context = Graph.from_dict(hyper_edge_set_dict={"line": line, "bus": bus_context}, n_addresses=n)

        # Oracle
        # Use bus for the solution (phase angles)
        bus_oracle = HyperEdgeSet.from_dict(port_dict=None, feature_dict={"phase_angle": theta})
        oracle = Graph.from_dict(hyper_edge_set_dict={"bus": bus_oracle}, n_addresses=n)

        return context, oracle

    def generate_problem(self) -> LinearSystemProblem:
        context, oracle = self._generate_numpy_problem()
        return LinearSystemProblem(context=Graph.to_jax_backend(context), oracle=Graph.to_jax_backend(oracle))

    def generate_problem_batch(self, batch_size: int = 8) -> LinearSystemProblemBatch:

        context_list, oracle_list = [], []

        for _ in range(batch_size):
            context, oracle = self._generate_numpy_problem()
            context_list.append(context)
            oracle_list.append(oracle)

        max_context_shape = GraphShape(
            hyper_edge_sets={
                "line": np.array(self.n_max * (self.n_max - 1) // 2),
                "bus": np.array(self.n_max),
            },
            addresses=np.array(self.n_max),
        )
        max_oracle_shape = GraphShape(hyper_edge_sets={"bus": np.array(self.n_max)}, addresses=np.array(self.n_max))

        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]

        # Pad and collate on NumPy, then transfer the whole batch to device in one go.
        context_batch = Graph.to_jax_backend(collate_graphs(context_list))
        oracle_batch = Graph.to_jax_backend(collate_graphs(oracle_list))

        return LinearSystemProblemBatch(context=context_batch, oracle=oracle_batch)


class LinearSystemProblemLoader(ProblemLoader):
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
