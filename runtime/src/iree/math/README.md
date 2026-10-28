# IREE Deterministic Math

`iree/math` provides compact numerical primitives whose result mappings can be
shared by the runtime and compiler. It is not a host `libm` wrapper: each
public function names one stable operation, format, and numerical contract so
constant evaluation and dynamic execution can agree across hosts.

Ordinary F32/F64 square root, reciprocal square root, and natural/base-two
exponentials preserve gradual underflow. Both scalar and vector constant
folding and the Core VM call these same typed functions:

```c
float root = iree_math_sqrt_f32(x);
double weight = iree_math_exp_f64(score);
double power = iree_math_exp2_f64(exponent);
```

Square root is correctly rounded and uses hardware instructions when available.
An integer restoring algorithm supplies the same payloads on hardware without
square root. Reciprocal square root rounds after both square root and division.
The exponentials use Arm's table-and-polynomial algorithms with fixed unfused
rounding points and at most one adjacent value of error from correct rounding.
Their exceptional-value behavior and arithmetic profile are part of their
stable mapping, not host library policy.

Fused multiply-add rounds the exact product and addend once. F32/F64 use a
native instruction when enabled by the build ISA, with an owned software
implementation otherwise. Narrow formats share an intermediate that preserves
the side of their final rounding midpoint, including tiny addends lost by
ordinary F64 addition. Compiler folding uses all three paths, and F32/F64 VM
instructions use the same typed functions. Neither depends on a platform `fma`
implementation.

The f32 approximation operations model machine-shaped leaves used by ML target
recipes:

```c
float y0 = iree_math_exp2_f32_approx(x);
float y1 = iree_math_log2_f32_approx(x);
float y2 = iree_math_reciprocal_f32_approx(x);
float y3 = iree_math_rsqrt_f32_approx(x);
float y4 = iree_math_sqrt_f32_approx(x);
float y5 = iree_math_sin_turns_f32_approx(x);
float y6 = iree_math_cos_turns_f32_approx(x);
```

`approx` does not grant the implementation a tolerance. It identifies one
frozen IREE result mapping selected for a machine-like accuracy and denormal
contract. Changing any result payload requires a new named contract and, when
encoded in bytecode, a new selector. The public headers document each mapping's
NaN, infinity, signed-zero, denormal, and accuracy behavior.

## Execution Profile

Leaf functions contain no allocation, status handling, synchronization,
`errno`, or floating-environment transitions. The caller establishes the
documented non-trapping rounding and denormal profile once around its execution
scope. This lets a VM invocation or compiler evaluation segment pay for at most
one host FPU transition instead of one transition per scalar operation.

Package-owned compile options prevent an embedding application's unrestricted
fast-math flags from changing the frozen mappings. Implicit contraction is
disabled; fused arithmetic, where required by an algorithm, must be explicit.
Clang evaluates intermediates at their declared source width, including when
an embedding application requests wider expression evaluation globally.
The ordinary mappings require
gradual underflow, while approximation leaves own their documented flush rules.

## Organization

Operations are grouped by algorithm family and provenance:

- `exponential.h` owns natural/base-two exponential and base-two logarithm.
- `fma.h` owns fused arithmetic and narrow-format intermediate evaluation.
- `roots.h` owns reciprocal, reciprocal square root, and square root.
- `trigonometry.h` owns turns-based sine and cosine.

All clusters form one `:math` build target. Keeping their implementations in
separate translation units preserves readable ownership and ordinary
dead-stripping without multiplying build graph actions. `float_bits.h` is a
private representation helper, not another public API.

New operations require a concrete compiler or runtime consumer, an explicit
payload contract, independent accuracy evidence, cross-platform bit vectors,
and measured latency and closure size. Target-specific approximations belong
with their target unless they are selected as the portable IREE mapping.
