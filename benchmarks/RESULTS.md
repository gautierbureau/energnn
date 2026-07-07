# GPU benchmark results

Measured with `benchmarks/parallel_gpu_matrix.py` on the integration branch
(`claude/perf-gpu-integration-tim3zs`).

**Hardware**: NVIDIA RTX A1000 6GB Laptop GPU (Ampere, CUDA 13.1 driver) ·
**Software**: JAX 0.9.0 (`jax[cuda12]`), single device ·
**Model**: latent 16, hidden [32], 20 message-passing steps, `normalizer_external_updates=True`.
Steady-state ms/step after a compile warm-up epoch; `score` is a validation-convergence spot check.

## n_max=64, batch_size=64

| batching | dtype | mlps     | ms/step | score           |
|----------|-------|----------|--------:|-----------------|
| dense    | fp32  | per-port |  168.85 | 9.884 → 0.3612  |
| dense    | fp32  | fused    |  190.78 | 9.884 → 0.3612  |
| dense    | bf16  | per-port |  121.22 | 9.846 → 0.3612  |
| dense    | bf16  | fused    |  143.24 | 9.846 → 0.3613  |
| union    | fp32  | per-port |   70.76 | 15.174 → 0.8512 |
| union    | fp32  | fused    |   73.17 | 15.174 → 0.8512 |
| union    | bf16  | per-port |   64.04 | 15.115 → 0.8516 |
| union    | bf16  | fused    |   66.18 | 15.115 → 0.8517 |

## n_max=128, batch_size=32

| batching | dtype | mlps     | ms/step | score            |
|----------|-------|----------|--------:|------------------|
| dense    | fp32  | per-port |  273.59 | 57.657 → 0.4285  |
| dense    | fp32  | fused    | FAILED¹ | —                |
| dense    | bf16  | per-port |  209.28 | 57.425 → 0.4285  |
| dense    | bf16  | fused    |  216.99 | 57.425 → 0.4284  |
| union    | fp32  | per-port |  109.03 | 74.045 → 0.8804  |
| union    | fp32  | fused    |  109.55 | 74.045 → 0.8804  |
| union    | bf16  | per-port |   91.06 | 73.772 → 0.8806  |
| union    | bf16  | fused    |   84.62 | 73.772 → 0.8806  |

¹ `XlaRuntimeError: INTERNAL: Autotuning failed` on a `f32[20,2,32,8128,32]`
backward transpose (~5 GB): the fused einsum's VJP under `vmap`+`scan`
materializes a stacked per-port intermediate that explodes on the padded dense
batch. The same fusion is fine on the union path (smaller true shapes, no vmap).

## Verdicts

- **Union batching (PR #7): clear GPU win.** 2.4× at n_max=64, 2.5× at
  n_max=128 versus dense — larger than the 1.9× measured on CPU, and growing
  with size heterogeneity. Validated as the recommended batching mode.
- **bf16 `compute_dtype` (PR #9): GPU win, convergence parity.** 1.4× on dense
  and 1.1–1.2× on union (kernels there are smaller), with validation scores
  matching fp32 to 3 decimals in every configuration. It *lost* on CPU
  (emulated), so keep it opt-in and enable on GPU/TPU runs.
- **Fused port MLPs (PR #8): keep off by default.** Slower in every dense
  configuration (and *fails* at scale on dense — see ¹), neutral on union fp32,
  and only a modest 1.08× on union+bf16 at n_max=128. The flag stays available
  (`fuse_port_mlps=True`) for future hardware, but the evidence does not
  support enabling it, and PR #8's standalone (always-on) form should be closed
  in favour of the flag in this branch.
- **Combined effect**: the previous default configuration (dense, fp32) at
  n_max=128 runs at 273.6 ms/step; the recommended configuration
  (union + bf16) runs at 91.1 ms/step — **3.0× faster end to end** on the same
  hardware, with unchanged convergence.

## GPU portability note

On Ampere, JAX float32 matmuls default to TF32 (~1e-4 relative error). One
CPU-calibrated exact-equivalence test needed `jax.default_matmul_precision("highest")`
to pass on GPU (fixed on this branch); tolerance-based tests were unaffected.
