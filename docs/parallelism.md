# Distributed & Parallel Training

This document describes how EnerGNN parallelises training across multiple devices
(GPUs/TPUs on one machine) and multiple hosts (a cluster of machines): the
strategy it uses, *why* that strategy fits this model, and the alternatives that
were considered.

- [The strategy we use: data parallelism over disjoint unions](#the-strategy-we-use-data-parallelism-over-disjoint-unions)
- [Single host, multiple devices](#single-host-multiple-devices)
- [Multiple hosts (a cluster)](#multiple-hosts-a-cluster)
- [Why this strategy (and not the alternatives)](#why-this-strategy-and-not-the-alternatives)

## The strategy we use: data parallelism over disjoint unions

EnerGNN uses **data parallelism**: the model parameters are *replicated* on every
device, each device processes a different slice of the batch, and the gradients
are averaged across devices with an all-reduce before the optimizer step. There
is **no model parallelism** — the parameters are small class-specific MLPs
(kilobytes), so splitting them across devices would add communication for no
memory benefit.

What makes EnerGNN's data parallelism unusually clean is *how the batch is split*.
Instead of padding every graph to a common shape and stacking them on a batch
axis, a group of graphs is concatenated into a single **disjoint-union graph**
with offset addresses (see `energnn.graph.union_graphs`). Because every model
stage is *local* — per-hyper-edge MLPs, and scatter/gather message passing keyed
by global addresses — no port of one instance ever references another instance's
addresses. Message passing therefore **cannot cross instance boundaries**, and by
extension it cannot cross device boundaries when each device holds whole
instances.

The consequence is the key property of this design:

> **The forward pass is communication-free.**
> With one disjoint union per device, the compiled forward (and its backward)
> contains **zero collectives** — no all-gather, no all-to-all, no halo exchange.
> The *only* collective in a training step is the gradient all-reduce over the
> (tiny, replicated) parameters, which JAX/XLA inserts automatically.

This is what lets the same code scale from one device to many devices to many
machines with almost no additional machinery: the compute was already
embarrassingly parallel, so only the *host boundaries* (getting data in, getting
metrics out) needed work.

## Single host, multiple devices

Build a 1-D device mesh and pass it to the `Trainer`; ask the loader to pack the
batch into one union per device:

```python
import jax, numpy as np, optax
from jax.sharding import Mesh
from energnn.problem.example import LinearSystemProblemLoader
from energnn.model.ready_to_use import SmallRecurrentEquivariantGNN
from energnn.trainer import Trainer

n_devices = jax.device_count()
mesh = Mesh(np.array(jax.devices()), ("data",))

loader = LinearSystemProblemLoader(
    mode="union", n_shards=n_devices, batch_size=n_devices * 8, mesh=mesh,
)
model = SmallRecurrentEquivariantGNN(
    in_structure=loader.context_structure, out_structure=loader.decision_structure,
)
trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3), mesh=mesh)
trainer.train(train_loader=loader, n_epochs=10)
```

What the `mesh` argument does:

- The context batch is sharded along its leading (batch) axis with
  `NamedSharding(mesh, P("data"))`. Every leaf of a collated batched-union `Graph`
  — features, ports, masks, shapes, segment ids — is batched on axis 0, so a
  single sharding covers the whole pytree.
- Parameters and optimizer state are replicated on the mesh, so they are
  compatible with the sharded input under `jit`.
- Because computation follows data, GSPMD partitions the existing vmapped forward
  along the batch axis and inserts the gradient all-reduce — no change to the
  jitted step functions.

## Multiple hosts (a cluster)

Multi-host uses JAX's multi-controller model: one process per host, each driving
its local devices, all sharing one global mesh. The compute is identical to the
single-host case (still communication-free); only two host boundaries differ —
**data ingestion** and **metric aggregation** — plus making parameters global
replicated arrays. Launch one process per host (SLURM, `torchrun`-style, etc.),
each running:

```python
import jax, numpy as np, optax
from jax.sharding import Mesh
from energnn.problem.example import LinearSystemProblemLoader
from energnn.model.ready_to_use import SmallRecurrentEquivariantGNN
from energnn.trainer import Trainer

jax.distributed.initialize()                      # picks up the cluster env
mesh = Mesh(np.array(jax.devices()), ("data",))   # all devices across all hosts

loader = LinearSystemProblemLoader(
    mode="union",
    n_shards=jax.device_count(),                  # global device count
    batch_size=jax.device_count() * 8,
    process_count=jax.process_count(),
    process_index=jax.process_index(),
    mesh=mesh,
)
model = SmallRecurrentEquivariantGNN(
    in_structure=loader.context_structure, out_structure=loader.decision_structure,
    normalizer_external_updates=True,             # REQUIRED, see below
)
trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3), mesh=mesh)
trainer.train(train_loader=loader, n_epochs=10)
```

How the host boundaries are handled (all in `energnn.parallel`, and all no-ops
when `jax.process_count() == 1`):

- **Data ingestion.** Each process *deterministically generates the whole global
  batch* (same seed → same problems), computes identical shard budgets, and keeps
  only its `n_shards / process_count` shards. Those process-local shards are
  assembled into global `jax.Array`s with `jax.make_array_from_process_local_data`
  (`assemble_global`). No cross-process communication happens during loading.
- **Parameter replication.** `replicate` places parameters and optimizer state as
  fully-replicated global arrays on the mesh. It reads each leaf via
  `local_replica` (see below) rather than `jax.device_put` to a replicated
  sharding, because the latter runs a cross-process equality assert that
  nan-initialized T-Digest state would fail.
- **Metric aggregation.** Per-instance validation scores are sharded across
  processes, so `gather_to_host` (`process_allgather`) collects them before the
  host-side reduction.

> **Reading global arrays on the host.**
> A value that lives on a *global* `jax.Array` cannot be pulled to the host with
> `np.asarray` if the array spans another process's devices — even a *replicated*
> array is not "fully addressable" across processes. EnerGNN reads such values
> through two helpers: `gather_to_host` (for **sharded** data, does a
> `process_allgather`) and `local_replica` (for **replicated** data, reads this
> process's local copy). This is why host-side T-Digest ingestion works under
> multi-host.

> ⚠️ **Requirement: a callback-free forward.**
> Host callbacks (e.g. the default `TDigestNormalizer`'s `io_callback`)
> **deadlock inside a multi-host SPMD program**. Multi-host training therefore
> requires `normalizer_external_updates=True`, which moves the T-Digest update to
> the host and keeps the compiled forward callback-free. The `Trainer` raises a
> clear error if a genuine multi-host mesh is paired with a host-callback
> normalizer, rather than hanging.

> ⚠️ **Environment gotcha.**
> JAX's distributed initialisation and Gloo collectives can **hang** if HTTP/S
> proxy environment variables are set. Unset all `*PROXY*` variables (and
> `NO_PROXY`) in the launch environment for the training processes.

## Why this strategy (and not the alternatives)

**Dense (pad + vmap) data parallelism.** The simplest option shards the padded,
stacked batch axis. It works and needs no union machinery, but it inherits dense
batching's padding waste (compute scales as `batch × max_size`), and with
heterogeneous grid sizes most of the work is padding — multiplied across every
device. Union batching makes compute scale with the *sum* of instance sizes, so it
is strictly better here and was chosen as the default sharded layout.

**Model / tensor parallelism.** Splits individual layers across devices. Pointless
for EnerGNN: the parameters are tiny, so there is nothing to gain in memory and
only communication to lose.

**Single-graph spatial partitioning.** If one instance is too large to fit on a
single device (a continental-scale grid), the graph itself must be partitioned
(e.g. a METIS min-cut over addresses) with a **halo exchange** of boundary
addresses at every message-passing step. This is real distributed GNN territory
and involves cross-device collectives *inside* the forward — the opposite of the
communication-free property above. It is deliberately **not** implemented: it is
weeks of work, its performance hinges on partition quality, and it is only needed
when a single instance exceeds device memory. Until then, per-device disjoint
unions give linear scaling with no in-forward communication.

**Pipeline parallelism.** Splits the layer stack into stages across devices.
Irrelevant here — the model is shallow and the bottleneck is data volume, not
depth.

---

**Summary of the trade-off EnerGNN takes:** it assumes **individual instances fit
on one device** (true for realistic energy networks) and optimises the common
case — many heterogeneously-sized instances — with disjoint-union data parallelism
that is communication-free in the forward and therefore scales cleanly from one
device to a multi-host cluster. Spatial partitioning remains the escape hatch for
the day a single instance no longer fits.
