# Loom Authoring Corpus

This corpus is the checked reference shape for hand-authored model and kernel
Loom. The files are ordinary `.loom` programs a model-porting agent should be
able to read, copy from, and extend: helper decomposition, provider selection,
local transform intent, correctness policy, and benchmark rows live in source
without a Python script generating the source itself.

The examples are tested through production-facing tools:

- `iree-benchmark-loom --dry_run` proves `check.case` and `check.benchmark`
  planning without requiring a local GPU during host-only CI.
- `iree-benchmark-loom --device=amdgpu --measure=dispatch_complete` compiles,
  executes correctness samples, and benchmarks the same sources on AMDGPU test
  hosts.

The timing flags used by the Bazel AMDGPU smoke targets are harness policy. The
source files name workloads and correctness expectations; iteration counts,
warmups, profiling, compile-time measurement, soak runs, and quick proof runs
belong to `iree-benchmark-loom` flags or embedding APIs.

## Source Map

| Pattern | Checked source |
| --- | --- |
| Concrete helper call | `ffn_gate_up_swiglu_q6q8.loom` calls `@q6_signed_pack_dot4i` with `func.call` because it is exact bit manipulation. |
| Template provider selection | `ffn_gate_up_swiglu_q6q8.loom` applies `model.q6q8.accumulate_part` so libraries can provide alternate packed-dot implementations. |
| Local unroll intent | `ffn_gate_up_swiglu_q6q8.loom` keeps block/part loops structured and marks the tiny trip-count loops with `unroll`. |
| Logical indexing | Both files use index/view math for logical rows, blocks, lanes, and dense tensor coordinates. |
| Dynamic case parameters | `mlp_down_projection_residual_bf16.loom` names `rows` on a `check.param.choice` and threads it through shapes, launch geometry, and the kernel ABI. |
| Benchmark slices | `mlp_down_projection_residual_bf16.loom` has an anonymous full sweep plus named decode/full rows with assignment dictionaries. |

## FFN q6/q8 Gate-Up SwiGLU

`ffn_gate_up_swiglu_q6q8.loom` models a q6_K-weight by q8_1 activation gate/up
fusion. The file keeps the model structure visible: launch topology, q8 input
views, q6 gate/up weight views, shared q8 load, gate/up accumulation, subgroup
reduction, SiLU, and final store.

The q6 sign-pack helper is a direct `func.call` because the call site wants that
specific bit helper. The q6/q8 part accumulator is a `func.template` provider
for the `model.q6q8.accumulate_part` contract and the kernel uses
`func.apply<model.q6q8.accumulate_part>`. Selection rewrites the apply to an
inline call to the selected provider, then normal callable inlining removes the
boundary before executable lowering. That is the intended library shape for
layout, target, or algorithm families: the model kernel asks for a contract,
while libraries own provider symbols and selection predicates.

The block/part loops are source-level transform intent, not source expansion.
They carry gate/up accumulators as loop results and request local unrolling only
where the q6_K trip counts are tiny and compile-time known. This keeps the
logical reduction visible to analysis while still producing the expanded low
code expected by the backend.

The zero case is an execution smoke test: zero weights and activations make the
expected tensor simple, while the dispatch still exercises unpack, dot, scale,
reduction, SiLU, and store. Higher-fidelity math oracles belong in the external
fixture/reference layer when the expected values are too large or too expensive
to express inline.

## MLP Down-Projection Residual

`mlp_down_projection_residual_bf16.loom` keeps one down-projection kernel with a
residual add and a named `rows` parameter. The parameter drives the case tensor
shapes, the scalar kernel argument, dynamic buffer views, and dispatch geometry,
so one authored source covers both a two-row decode-shaped sample and the full
projection.

The anonymous `check.benchmark<@mlp_down_projection_residual_case>` sweeps all
case samples and receives generated benchmark names. The named benchmark rows
pin specific samples with assignment dictionaries:

```loom
check.benchmark<@mlp_down_projection_residual_case> @mlp_down_projection_residual_decode {rows = 2}
check.benchmark<@mlp_down_projection_residual_case> @mlp_down_projection_residual_full {rows = 3584}
```

The case uses deterministic iota inputs and zero projection weights for a
residual-preservation oracle. The AMDGPU dispatch test runs this file with
per-sample compilation so each selected row count becomes a compile-time fact
before launch geometry and memory legality are finalized.

## Authoring Rules

`func.call` is an exact symbol reference. It is the right spelling when the
caller wants one specific helper or declaration, as with
`@q6_signed_pack_dot4i` and `@bf16_dot32`.

`func.apply<K>` is a compile-time implementation demand. The key is a contract,
not a symbol. The selection pass resolves live applies against visible
`func.template` providers, prunes dead private providers, rewrites selected
sites to inline calls, and leaves normal inlining to splice the body. Good
contract names describe a reusable motif or layout operation rather than a
particular model brand.

`inline` and `noinline` describe callable-boundary intent. `inline` is useful
for private helpers that should disappear before target-sensitive lowering.
`noinline` preserves a real callable boundary and therefore needs a target-ready
symbol when target lowering reaches it. `hot` and `cold` are separate
temperature hints for cost models or profile feedback; they are not substitutes
for authored inline policy.

Facts are source facts, not global flags. The examples use `index.assume`,
function signatures, named shape dimensions, and `check.param.choice` samples to
make dimensions and alignment available to passes. Module-level configuration
is reserved for values that are genuinely shared by the module or library.

`check.case` owns correctness policy for a workload. It creates inputs, calls
the unit under test, and states expectations. `check.benchmark<@case>` selects
which case samples should be timed. Benchmark rows name workloads; the runner
chooses timing rigor, output format, profiling, compile-time measurement, and
batching.

Python, C, and C++ remain appropriate for oracle code, external comparison
harnesses, fixture extraction, and binary fixture preparation. They are outside
the authored Loom source contract unless they are producing data consumed by a
checked `.loom` case.

## Failure Signals

| Signal | Mechanism to inspect |
| --- | --- |
| Residual `func.apply` after final selection | No provider implemented the contract, every provider was rejected by signature or predicates, or multiple highest-priority providers tied. |
| Template ambiguity | Matching providers need distinct priorities, sharper predicates, or separate contract keys. |
| Unresolved unroll intent | The loop trip count or requested factor was not known where the unroller ran; add facts earlier or leave the loop structured. |
| Inline/noinline conflict | Caller and callee policy disagree about whether the boundary may survive lowering. Fix the authored policy instead of relying on pass ordering. |
| Targetless helper reached target lowering | The pipeline missed callable specialization or inlining for the selected call graph. Adding target attributes to every reusable helper is the wrong source shape. |
| Benchmark parameter mismatch | A benchmark assignment dictionary referenced a name that is not a named `check.param` in the selected case. |

## Library-Scale Pressure

The q6/q8 accumulator intentionally exposes a wide signature: today it is the
honest Loom spelling for the view bundle and derived lane/block coordinates the
kernel needs. If several kernels repeat the same bundle, the pressure is for a
real representation primitive or a sharper helper boundary, not a source
generator.

The benchmark style scales by absence. A model library can carry many
`check.benchmark<@case>` rows because each row is just a workload selection.
Repeated timing dictionaries, profiling flags, and per-row harness policy would
make the source noisy and would couple authored IR to one command-line tool.
