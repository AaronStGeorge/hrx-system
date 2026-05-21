# Loom Authoring Corpus

This corpus contains authored model-kernel examples. The files here are meant
to look like source a model-porting agent would maintain by hand: helper
decomposition, local transform intent, correctness cases, and benchmark records
are all in the `.loom` file.

The examples are tested through production-facing tools:

- `iree-tune-loom --dry_run` proves `check.case` and `check.benchmark` planning
  without requiring a local GPU during host-only CI.
- `iree-tune-loom --device=amdgpu --measure=dispatch_complete` compiles,
  executes correctness samples, and benchmarks the same sources on AMDGPU test
  hosts.

## FFN q6/q8 Gate-Up SwiGLU

`ffn_gate_up_swiglu_q6q8.loom` models a q6_K-weight by q8_1 activation
gate/up fusion. The file keeps the model structure visible: a concrete q6
sign-pack helper, one shared gate-or-up accumulate helper, result-carrying
block/part loops with local unroll intent, subgroup reduction, SiLU, and a
single dispatch-shaped zero case with one benchmark row.

The zero case is intentionally structured as an execution smoke test: zero
weights and activations make the expected tensor simple, while the dispatch
still exercises the q6/q8 unpack, dot, scale, reduction, SiLU, and store path.

## MLP Down-Projection Residual

`mlp_down_projection_residual_bf16.loom` keeps one down-projection kernel with a
residual add and a named `rows` parameter. The parameter drives the case tensor
shapes, the scalar kernel argument, dynamic buffer views, and dispatch geometry
so one authored source covers both a two-row decode-shaped sample and the full
projection. The anonymous benchmark sweeps both row counts, while named
benchmark rows pin the decode and full samples for targeted runs.

The case uses deterministic iota inputs and zero projection weights for a simple
residual-preservation oracle. The AMDGPU dispatch test runs this file with
per-sample compilation so each selected row count becomes a compile-time fact
before launch geometry and memory legality are finalized.

Iteration counts, warmups, profiling, compile timing, and soak/quick modes live
on `iree-tune-loom` flags or embedding APIs.
