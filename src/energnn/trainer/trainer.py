# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import time
from typing import Literal

import flatdict
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from optax import GradientTransformation
from orbax.checkpoint import CheckpointManager
from tqdm import tqdm

from energnn.graph import Graph
from energnn.model import GNN
from energnn.problem import ProblemBatch, ProblemLoader
from energnn.tracker import Tracker
from .utils import TaskLogger

# Types
GraphBatch = Graph

logger = logging.getLogger(__name__)


def _cast_cotangent_to_primal_dtype(cotangent_pytree, primal_pytree):
    """
    Cast each leaf in `cotangent_pytree` to the dtype of the corresponding leaf in `primal_pytree`.
    Leaves that don't appear to have a .dtype are returned unchanged.
    """

    def _cast_leaf(c, p):
        try:
            target_dtype = p.dtype
        except Exception:
            # Keep the original cotangent leaf if we cannot read dtype
            return c
        return jnp.asarray(c, dtype=target_dtype)

    return jax.tree.map(_cast_leaf, cotangent_pytree, primal_pytree)


def _functional_update(opt_graphdef, opt_state, model_graphdef, params, rest, gradient):
    """Functional optimizer update, designed to be JIT-compiled once and reused.

    Operates on split pytrees instead of NNX modules: passing the modules through
    ``nnx.jit`` would re-flatten the model and optimizer graphs on every call, which
    dominates the host-side cost of a training step. The parameter and optimizer-state
    buffers are donated by the caller, so the update happens in place on device.
    """
    optimizer = nnx.merge(opt_graphdef, opt_state)
    model = nnx.merge(model_graphdef, params, rest)
    optimizer.update(model, gradient)
    _, new_params, _ = nnx.split(model, nnx.Param, ...)
    _, new_opt_state = nnx.split(optimizer)
    return new_params, new_opt_state


def _setup_ckpt_mngr(checkpoint_manager: CheckpointManager, optim_mode: Literal["minimize", "maximize"]):
    checkpoint_manager._options.best_fn = lambda x: x["score"]
    if optim_mode == "minimize":
        checkpoint_manager._options.best_mode = "min"
    elif optim_mode == "maximize":
        checkpoint_manager._options.best_mode = "max"


class Trainer:
    r"""
    Trainer implementation.

    This basic trainer relies on the training of a permutation-equivariant
    Graph Neural Network :math:`\hat{y}_\theta` over a dataset of problem instances.
    For a fixed problem instance with objective function :math:`f`
    and context :math:`x`, the parameter :math:`\theta` is updated according to the following gradient descent step,

    .. math::
        \theta \gets \theta - \alpha . J_\theta[\hat{y}_\theta](x)^\top .
        \nabla_y f (\hat{y}_\theta(x);x),

    where :math:`J_\theta[\hat{y}_\theta]` is the Jacobian matrix of the GNN
    :math:`\hat{y}_\theta`, and :math:`\nabla_y f` is the gradient of the
    objective function :math:`f` *w.r.t* the decision :math:`y`.
    For the sake of readability, a basic gradient descent is used --
    with a learning rate :math:`\alpha` --
    but more complex optimizers are possible.

    After every training epoch, the current trainer is checkpointed.

    :param model: Core Graph Neural Network model.
    :type model: GNN
    :param gradient_transformation: Optax gradient transformation.
    :type gradient_transformation: optax.GradientTransformation
    """

    def __init__(
        self,
        *,
        model: GNN,
        gradient_transformation: GradientTransformation,
        mesh: jax.sharding.Mesh | None = None,
    ):
        """
        :param model: Core Graph Neural Network model.
        :param gradient_transformation: Optax gradient transformation.
        :param mesh: Optional 1-D device mesh for data-parallel training. When provided,
            every context batch is sharded along its leading (batch) axis across the
            mesh before the forward pass, parameters stay replicated, and XLA inserts
            the gradient all-reduce automatically. The batch axis must be divisible by
            the number of devices. Pairs naturally with batches of per-device disjoint
            unions (see :func:`~energnn.graph.union_graphs`), where message passing
            needs no cross-device communication at all.
        """
        self.model: GNN = model
        self.optimizer = nnx.Optimizer(self.model, gradient_transformation, wrt=nnx.Param)
        self.train_step: int = 0
        self.best_score: float | None = None
        self.mesh = mesh
        if mesh is not None:
            if len(mesh.axis_names) != 1:
                raise ValueError(f"Trainer expects a 1-D data-parallel mesh, got axes {mesh.axis_names}.")
            self._data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(mesh.axis_names[0]))
        else:
            self._data_sharding = None

        # Cache JIT-compiled wrappers to avoid NNX re-tracing overhead each step.
        # `get_info` is static because downstream code branches on its concrete value.
        self._jit_apply = nnx.jit(self._apply_forward_vjp, static_argnames=("get_info",))
        self._jit_backward = jax.jit(self._backward_from_vjp)
        self._jit_eval_forward = nnx.jit(self._eval_forward, static_argnames=("get_info",))
        # Parameter and optimizer-state buffers are donated so XLA can update them
        # in place instead of allocating fresh buffers.
        self._jit_update_params = nnx.jit(_functional_update, donate_argnums=(1, 3))

    @staticmethod
    def _apply_forward_vjp(graphdef, params, rest, jax_context, get_info):
        """Forward pass + VJP setup, designed to be JIT-compiled once and reused."""

        def f_forward(p, r):
            model = nnx.merge(graphdef, p, r)
            decision, _ = model.forward_batch(graph=jax_context, get_info=get_info)
            _, _, r_updated = nnx.split(model, nnx.Param, ...)
            return decision, r_updated

        (jax_decision, rest_updated), vjp_fn = jax.vjp(f_forward, params, rest)
        return jax_decision, rest_updated, vjp_fn

    @staticmethod
    def _backward_from_vjp(vjp_fn, jax_gradient, jax_decision, rest_updated):
        """Compiled backward pass, designed to be JIT-compiled once and reused.

        Calling the ``vjp_fn`` returned by :meth:`_apply_forward_vjp` directly would
        interpret the transposed computation op-by-op; ``vjp_fn`` is a pytree
        (``jax.tree_util.Partial``) with a stable structure across steps, so passing it
        through ``jax.jit`` compiles the whole backward pass once. The cotangent dtype
        cast and the zero cotangents for the non-parameter state are folded in as well.
        """
        jax_cotangent = _cast_cotangent_to_primal_dtype(jax_gradient, jax_decision)
        rest_cotangent = jax.tree.map(jnp.zeros_like, rest_updated)
        (grads_params, _) = vjp_fn((jax_cotangent, rest_cotangent))
        return grads_params

    @staticmethod
    def _eval_forward(model, context, get_info: bool = True):
        """Forward pass for evaluation, designed to be JIT-compiled once and reused."""
        decision, info = model.forward_batch(graph=context, get_info=get_info)
        _, _, r_updated = nnx.split(model, nnx.Param, ...)
        return decision, info, r_updated

    def train(
        self,
        *,
        train_loader: ProblemLoader,
        val_loader: ProblemLoader | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        n_epochs: int,
        tracker: Tracker | None = None,
        log_period: int | None = None,
        eval_period: int | None = None,
        eval_before_training: bool = False,
        eval_after_epoch: bool = True,
        progress_bar: bool = True,
        optim_mode: Literal["minimize", "maximize"] = "minimize",
    ) -> float:
        r"""
        Trains the model over the train loader, periodically validates the model, tracks metrics, and checkpoints the model.

        :param train_loader: Problem loader used for training.
        :param val_loader: Problem loader used for validation.
        :param checkpoint_manager: Checkpoint manager for saving checkpoints.
        :param n_epochs: Number of training epochs to perform.
        :param tracker: Experiment tracker.
        :param log_period: Number of training iterations between two logs, None for no logs.
        :param eval_period: Number of training epochs between two evaluations, None for no evaluations.
        :param eval_before_training: If true, evaluate metrics over the full validation loader before training.
        :param eval_after_epoch: If true, evaluate metrics over the full validation loader after each epoch.
        :param progress_bar: If true, display a progress bar during training.
        :param optim_mode: Optimization mode, either "minimize" or "maximize". Overrides the checkpoint manager's `best_mode`.
        :return: Best average score obtained on the validation loader.
        """
        if checkpoint_manager is not None:
            _setup_ckpt_mngr(checkpoint_manager, optim_mode=optim_mode)

        # Evaluation over the full validation loader before training.
        if eval_before_training and (val_loader is not None):
            _ = self.run_evaluation(
                val_loader=val_loader,
                progress_bar=progress_bar,
                tracker=tracker,
                checkpoint_manager=checkpoint_manager,
                optim_mode=optim_mode,
                position=0,
            )

        for epoch in range(1, n_epochs + 1):

            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=True, unit="batch", disable=not progress_bar, position=0
            )
            for problem_batch in pbar:

                # Perform one training step
                if (log_period is not None) and (self.train_step % log_period == 0) and (tracker is not None):
                    infos = self.training_step(problem_batch, get_info=True)
                    tracker.run_append(infos={"train": infos}, step=self.train_step)
                else:
                    _ = self.training_step(problem_batch, get_info=False)

                # If True, run evaluation
                if (eval_period is not None) and (self.train_step % eval_period == 0) and (val_loader is not None):
                    _ = self.run_evaluation(
                        val_loader=val_loader,
                        progress_bar=progress_bar,
                        tracker=tracker,
                        checkpoint_manager=checkpoint_manager,
                        optim_mode=optim_mode,
                        position=0,
                    )

                self.train_step += 1

            # At the end of each epoch, save latest model and perform an evaluation, unless evaluation was just run.
            if (eval_period is not None) and (self.train_step % eval_period == 0):
                continue
            elif eval_after_epoch and (val_loader is not None):
                _ = self.run_evaluation(
                    val_loader=val_loader,
                    progress_bar=progress_bar,
                    tracker=tracker,
                    checkpoint_manager=checkpoint_manager,
                    optim_mode=optim_mode,
                    position=0,
                )

        if checkpoint_manager is not None:
            checkpoint_manager.wait_until_finished()
        return self.best_score

    def run_evaluation(
        self,
        *,
        val_loader,
        progress_bar: bool = True,
        tracker: Tracker = None,
        checkpoint_manager: CheckpointManager | None = None,
        optim_mode: Literal["minimize", "maximize"] = "minimize",
        position: int = 0,
    ) -> float:
        """
        Runs an evaluation and checkpoints.

        :param val_loader: Validation data loader.
        :param progress_bar: If true, display a progress bar during evaluation.
        :param tracker: Experiment tracker.
        :param checkpoint_manager: Checkpoint manager for saving checkpoints.
        :param position: Position of the progress bar if shown.
        :return: Average score obtained on the validation set.
        """
        self.model.eval()  # Set model to eval mode

        # Detailed per-feature statistics are only consumed by the tracker; skip
        # computing them (quantiles inside the jitted forward + host transfers)
        # when there is no tracker to report them to.
        mean_score, infos = self.eval(val_loader, progress_bar=progress_bar, position=position, get_info=tracker is not None)
        if self.best_score is None:
            self.best_score = mean_score
        else:
            if (optim_mode == "minimize") and (mean_score < self.best_score):
                self.best_score = mean_score
            elif (optim_mode == "maximize") and (mean_score > self.best_score):
                self.best_score = mean_score

        if tracker is not None:
            tracker.run_append(infos={"eval": infos}, step=self.train_step)

        if checkpoint_manager is not None:
            self.save_checkpoint(checkpoint_manager=checkpoint_manager, score=mean_score)

        return mean_score

    def save_checkpoint(self, *, checkpoint_manager: CheckpointManager, score: float) -> None:
        """Saves the current model and optimizer state as a checkpoint.

        :param checkpoint_manager: Checkpoint manager to use for saving the checkpoint.
        :param score: Mean score obtained on the validation set.

        Returns:
            str | None: Local path to the saved checkpoint directory,
             or None if the checkpoint manager did not save the checkpoint.
        """
        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        checkpoint_data = {
            "model": model_state,
            "optimizer": opt_state,
            "step": self.train_step,
            "score": score,
        }
        checkpoint_manager.save(self.train_step, args=ocp.args.Composite(default=ocp.args.StandardSave(checkpoint_data)))

    def load_checkpoint(self, checkpoint_manager: CheckpointManager, step: int | None = None, best: bool = False) -> None:
        """Loads a checkpoint from the checkpoint manager.

        :param checkpoint_manager: Checkpoint manager to use for loading the checkpoint.
        :param step: Step of the checkpoint to load. If None, load the latest checkpoint.
        :param best: If true, load the best checkpoint.
        """
        if best:
            step = checkpoint_manager.best_step()
        elif step is None:
            step = checkpoint_manager.latest_step()

        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        abstract_checkpoint_data = {"model": model_state, "optimizer": opt_state, "step": self.train_step, "score": 0.0}
        restored = checkpoint_manager.restore(
            step, args=ocp.args.Composite(default=ocp.args.StandardRestore(abstract_checkpoint_data))
        )
        restored = restored["default"]
        nnx.update(self.model, restored["model"])
        nnx.update(self.optimizer, restored["optimizer"])
        self.train_step = restored["step"]

    def eval(
        self, loader: ProblemLoader, progress_bar: bool = False, position: int = 0, get_info: bool = True
    ) -> tuple[float, dict]:
        """
        Evaluates the amortizer over a problem loader by averaging the score scalar.

        :param loader: Problem loader over which the amortizer is evaluated.
        :param progress_bar: If true, display a progress bar during evaluation.
        :param position: Position of the progress bar if shown.
        :param get_info: If true, collect detailed per-feature statistics for each batch.
            Disabling it skips the quantile computations and host transfers, which
            speeds up evaluation when only the score is needed.
        :return: Average score obtained over the problem loader.
        """
        score_list, infos_list = [], []
        # Running nan-aware mean, to avoid re-concatenating all previous scores at every batch.
        score_sum, score_count = 0.0, 0
        pbar = tqdm(loader, desc="Validation", unit="batch", leave=True, disable=not progress_bar, position=position)
        for step, problem_batch in enumerate(pbar):
            score_batch, info_batch = self.eval_step(step, problem_batch, get_info=get_info)
            score_list.append(score_batch)
            infos_list.append(info_batch)
            if progress_bar:
                score_array = np.asarray(score_batch, dtype=float)
                score_sum += np.nansum(score_array)
                score_count += int(np.sum(~np.isnan(score_array)))
                pbar.set_postfix(score=f"{score_sum / max(score_count, 1):.4e}")

        mean_score = np.nanmean(np.concatenate(score_list)).astype(float)

        # Concatenate all infos together.
        keys = set.union(*[set(info_batch.keys()) for info_batch in infos_list])
        infos = {}
        for k in keys:
            vals = [infos.get(k, np.array([])) for infos in infos_list]
            if any(np.ndim(v) == 0 for v in vals):
                infos[k] = np.stack(vals)
            else:
                infos[k] = np.concatenate(vals)
        infos["score"] = mean_score

        return mean_score, infos

    def training_step(self, problem_batch: ProblemBatch, get_info: bool) -> dict:
        """
        Performs a training step to update model parameters.

        :param problem_batch: A batch of problems for training.
        :param get_info: Whether to compute information or not.
        :return: A dictionary of information about the training step, or list of dictionaries.
        """
        # Fine-grained stage timing forces a host-device synchronization after every
        # stage (block_until_ready), which serializes the training loop. Only pay
        # that cost when INFO logging is actually enabled.
        log_timings = logger.isEnabledFor(logging.INFO)

        def _sync_and_log(stage: str, t_start: float, value=None) -> None:
            if log_timings:
                if value is not None:
                    jax.block_until_ready(value)
                logger.info(f"[training_step {self.train_step}] {stage}: {(time.perf_counter() - t_start) * 1000:.3f} ms")

        with TaskLogger(logger, f"Training step {self.train_step}"):

            t_start = time.perf_counter()
            self.model.train()  # Set model to train mode
            _sync_and_log("model.train()", t_start)

            infos = {}
            t_start = time.perf_counter()
            jax_context, infos["1_context"] = problem_batch.get_context(get_info=get_info, step=self.train_step)
            if self._data_sharding is not None:
                jax_context = jax.device_put(jax_context, self._data_sharding)
            _sync_and_log("get_context", t_start, jax_context)

            # Host-side normalizer statistics update (no-op unless the model's normalizer
            # is configured for external updates).
            t_start = time.perf_counter()
            ingest = getattr(self.model, "ingest", None)
            if ingest is not None:
                ingest(graph=jax_context)
            _sync_and_log("ingest", t_start)

            t_start = time.perf_counter()
            graphdef, params, rest = nnx.split(self.model, nnx.Param, ...)
            _sync_and_log("nnx.split", t_start)

            t_start = time.perf_counter()
            jax_decision, rest_updated, vjp_fn = self._jit_apply(graphdef, params, rest, jax_context, get_info)
            _sync_and_log("forward", t_start, jax_decision)

            t_start = time.perf_counter()
            nnx.update(self.model, rest_updated)
            _sync_and_log("nnx.update", t_start)

            t_start = time.perf_counter()
            jax_gradient, infos["3_gradient"] = problem_batch.get_gradient(
                decision=jax_decision, get_info=get_info, step=self.train_step
            )
            _sync_and_log("get_gradient", t_start, jax_gradient)

            # Backward pass (compiled; includes the cotangent dtype cast and the zero
            # cotangents for the non-parameter state)
            t_start = time.perf_counter()
            grads_params = self._jit_backward(vjp_fn, jax_gradient, jax_decision, rest_updated)
            _sync_and_log("backward", t_start, grads_params)

            t_start = time.perf_counter()
            opt_graphdef, opt_state = nnx.split(self.optimizer)
            new_params, new_opt_state = self._jit_update_params(
                opt_graphdef, opt_state, graphdef, params, rest_updated, grads_params
            )
            nnx.update(self.model, new_params)
            nnx.update(self.optimizer, new_opt_state)
            if log_timings:
                jax.block_until_ready(nnx.state(self.model))
            _sync_and_log("update_params", t_start)

            infos["4_update"] = {}

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return infos

    def eval_step(self, eval_step: int, problem_batch: ProblemBatch, get_info: bool = True) -> tuple[list[float], dict]:
        """Evaluates the current gnn over a batch of problems.

        :param eval_step: Index of the current evaluation step.
        :param problem_batch: A problem batch.
        :param get_info: If true, collect detailed per-feature statistics.
        :return: A batch of scores and a dictionary of batched information.
        """
        with TaskLogger(logger, f"Eval step {eval_step}"):
            infos = {}

            jax_context, infos["1_context"] = problem_batch.get_context(get_info=get_info, step=self.train_step)
            if self._data_sharding is not None:
                jax_context = jax.device_put(jax_context, self._data_sharding)

            jax_decision, infos["2_forward"], rest_updated = self._jit_eval_forward(
                model=self.model, context=jax_context, get_info=get_info
            )

            score, infos["3_score"] = problem_batch.get_score(decision=jax_decision, get_info=get_info, step=self.train_step)

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return score, infos
