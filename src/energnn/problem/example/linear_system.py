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
from energnn.graph import HYPER_EDGE_GRAPH_IDS, TRUE_SHAPES, Graph, GraphShape, HyperEdgeSet, collate_graphs, union_graphs
from energnn.parallel import assemble_global, data_sharding, gather_to_host
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
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`.

        For dense batches, the mean runs over each instance's padded rows (batch axis 1).
        For disjoint-union batches, per-instance means are computed with segment
        reductions over the real (non-fictitious) entries only. For *batched* unions
        (one union per device shard), per-shard scores are computed with a vmap and
        flattened back to the original instance order, dropping padded empty instances.
        """
        if self.oracle.segments is not None:
            segments = self.oracle.segments
            true_addresses = jnp.asarray(segments[TRUE_SHAPES].addresses)
            n_graphs = true_addresses.shape[-1]

            features = {k: hes.feature_array for k, hes in decision.hyper_edge_sets.items()}
            oracle_features = {k: self.oracle.hyper_edge_sets[k].feature_array for k in features}
            masks = {k: self.oracle.hyper_edge_sets[k].non_fictitious for k in features}
            ids = {k: segments[HYPER_EDGE_GRAPH_IDS][k] for k in features}

            def union_scores(features, oracle_features, masks, ids):
                numerator = jnp.zeros(n_graphs)
                denominator = jnp.zeros(n_graphs)
                for key in features:
                    error = jnp.sum(jnp.square(features[key] - oracle_features[key]), axis=-1)
                    numerator = numerator.at[ids[key].astype(int)].add(error * masks[key], mode="drop")
                    denominator = denominator.at[ids[key].astype(int)].add(masks[key] * features[key].shape[-1], mode="drop")
                return numerator / jnp.maximum(denominator, 1.0)

            if true_addresses.ndim == 2:
                # Batched unions (one per device shard): instances were packed
                # contiguously, so flattening restores the original order. The scores are
                # sharded across processes, so gather them to the host (a no-op on a
                # single process) before dropping the padded empty instance slots.
                per_slot = jax.vmap(union_scores)(features, oracle_features, masks, ids).reshape(-1)
                valid = jnp.reshape(true_addresses > 0, -1)
                objective = gather_to_host(per_slot)[gather_to_host(valid)]
            else:
                objective = gather_to_host(union_scores(features, oracle_features, masks, ids))
            return objective.tolist(), {}

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


def _next_power_of_two(n: int, minimum: int = 8) -> int:
    """Smallest power of two >= n (with a floor), used to bucket union batch budgets."""
    return max(minimum, 1 << (int(n) - 1).bit_length())


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

    # Add remaining m - (n-1) edges. Candidate edges are the empty upper-triangular
    # entries, enumerated in row-major order (same order as the previous Python loop).
    iu, ju = np.triu_indices(n, k=1)
    free = B[iu, ju] == 0
    possible_u, possible_v = iu[free], ju[free]

    if len(possible_u) > 0 and m > n - 1:
        n_extra = min(m - (n - 1), len(possible_u))
        idx = np.random.choice(len(possible_u), n_extra, replace=False)
        for i in idx:
            u, v = possible_u[i], possible_v[i]
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

    def generate_problem_batch(
        self,
        batch_size: int = 8,
        mode: str = "dense",
        n_shards: int = 1,
        process_count: int = 1,
        process_index: int = 0,
        mesh: "jax.sharding.Mesh | None" = None,
    ) -> LinearSystemProblemBatch:
        """Generate a batch of problems.

        :param batch_size: Number of problem instances in the (global) batch.
        :param mode: ``"dense"`` pads every instance to the maximum shape and stacks them
            along a batch axis (processed with vmap). ``"union"`` concatenates the
            unpadded instances into one disjoint-union graph padded to a power-of-two
            total-size bucket, so compute scales with the actual total size.
        :param n_shards: Union mode only. With ``n_shards > 1``, instances are packed
            contiguously into ``n_shards`` equal-budget unions collated along a leading
            axis — one union per device for data-parallel training (shard the batch
            with the trainer's ``mesh``). Ports never cross instances, so message
            passing needs no cross-device communication.
        :param process_count: Number of processes (hosts) in a multi-host run. Each
            process deterministically generates the same global batch and keeps only its
            ``n_shards / process_count`` shards, so budgets stay identical across
            processes with no communication. Union mode only.
        :param process_index: Index of this process in ``[0, process_count)``.
        :param mesh: Global device mesh. When provided, the process-local shards are
            assembled into globally-sharded arrays (one union per device). Required for
            ``process_count > 1``; on a single process it is equivalent to placing the
            batch with ``jax.device_put``.
        """
        if process_count < 1 or not (0 <= process_index < process_count):
            raise ValueError(f"Invalid process layout: process_count={process_count}, process_index={process_index}.")
        if process_count > 1 and mode != "union":
            raise ValueError("process_count > 1 requires mode='union'.")
        if process_count > 1 and n_shards % process_count != 0:
            raise ValueError(f"n_shards ({n_shards}) must be divisible by process_count ({process_count}).")

        context_list, oracle_list = [], []

        for _ in range(batch_size):
            context, oracle = self._generate_numpy_problem()
            context_list.append(context)
            oracle_list.append(oracle)

        if mode == "dense":
            if n_shards != 1 or process_count != 1:
                raise ValueError("dense mode does not support sharding (use mode='union').")
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
        elif mode == "union":
            if n_shards < 1:
                raise ValueError(f"n_shards must be >= 1, got {n_shards}")
            if batch_size < n_shards:
                raise ValueError(f"batch_size ({batch_size}) must be >= n_shards ({n_shards}).")
            # Pack instances contiguously into balanced shards (no shard is empty);
            # smaller shards are padded with empty instance slots up to the capacity.
            base, extra = divmod(batch_size, n_shards)
            sizes = [base + (1 if d < extra else 0) for d in range(n_shards)]
            offsets = np.concatenate([[0], np.cumsum(sizes)])
            capacity = sizes[0]
            context_chunks = [context_list[offsets[d] : offsets[d + 1]] for d in range(n_shards)]
            oracle_chunks = [oracle_list[offsets[d] : offsets[d + 1]] for d in range(n_shards)]

            # Bucket the union budgets to powers of two over the largest shard, so
            # training compiles a handful of shape variants instead of one per batch,
            # and all shards share one padded shape (required to collate them). The
            # budget spans every shard, so all processes derive the same shapes.
            def _budget(chunks, count):
                return _next_power_of_two(max(count(chunk) for chunk in chunks))

            line_budget = _budget(context_chunks, lambda c: sum(int(g.true_shape.hyper_edge_sets["line"]) for g in c))
            bus_budget = _budget(context_chunks, lambda c: sum(int(g.true_shape.hyper_edge_sets["bus"]) for g in c))
            address_budget = _budget(context_chunks, lambda c: sum(len(g.non_fictitious_addresses) for g in c))
            context_shape = GraphShape(
                hyper_edge_sets={"line": np.array(line_budget), "bus": np.array(bus_budget)},
                addresses=np.array(address_budget),
            )
            oracle_shape = GraphShape(hyper_edge_sets={"bus": np.array(bus_budget)}, addresses=np.array(address_budget))

            # Keep only the shards owned by this process (all shards when single-process).
            shards_per_process = n_shards // process_count
            shard_slice = slice(process_index * shards_per_process, (process_index + 1) * shards_per_process)
            local_context_chunks = context_chunks[shard_slice]
            local_oracle_chunks = oracle_chunks[shard_slice]

            context_unions = [union_graphs(c, target_shape=context_shape, n_graphs=capacity) for c in local_context_chunks]
            oracle_unions = [union_graphs(c, target_shape=oracle_shape, n_graphs=capacity) for c in local_oracle_chunks]

            local_context = context_unions[0] if len(context_unions) == 1 else collate_graphs(context_unions)
            local_oracle = oracle_unions[0] if len(oracle_unions) == 1 else collate_graphs(oracle_unions)

            if mesh is not None and (process_count > 1 or n_shards > 1):
                # Assemble the process-local shards into globally-sharded arrays
                # (convert to the JAX backend first so the assembled graph keeps it).
                sharding = data_sharding(mesh)
                context_batch = assemble_global(Graph.to_jax_backend(local_context), sharding)
                oracle_batch = assemble_global(Graph.to_jax_backend(local_oracle), sharding)
            else:
                context_batch = Graph.to_jax_backend(local_context)
                oracle_batch = Graph.to_jax_backend(local_oracle)
        else:
            raise ValueError(f"Unknown batching mode: {mode!r}, expected 'dense' or 'union'.")

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
        mode: str = "dense",
        n_shards: int = 1,
        process_count: int = 1,
        process_index: int = 0,
        mesh: "jax.sharding.Mesh | None" = None,
    ):
        if mode not in ("dense", "union"):
            raise ValueError(f"Unknown batching mode: {mode!r}, expected 'dense' or 'union'.")
        if n_shards != 1 and mode != "union":
            raise ValueError("n_shards > 1 requires mode='union'.")
        if process_count > 1 and mesh is None:
            raise ValueError("Multi-host loading (process_count > 1) requires a device mesh.")
        self.seed = seed
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.n_max = n_max
        self.shuffle = shuffle
        self.mode = mode
        self.n_shards = n_shards
        self.process_count = process_count
        self.process_index = process_index
        self.mesh = mesh
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
        batch = self.generator.generate_problem_batch(
            batch_size=n_batch,
            mode=self.mode,
            n_shards=self.n_shards,
            process_count=self.process_count,
            process_index=self.process_index,
            mesh=self.mesh,
        )
        return batch

    def __len__(self):
        return max(self.dataset_size // self.batch_size, 1)
