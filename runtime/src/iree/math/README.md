# IREE Deterministic Math

`iree/math` provides compact numerical primitives whose result mappings can be
shared by the runtime and compiler. It is not a host `libm` wrapper: each
public function names one stable operation, format, and numerical contract so
constant evaluation and dynamic execution can agree across hosts.

Ordinary F32/F64 square root, reciprocal square root, cube root, exponentials,
logarithms, real powers, hyperbolic functions, trigonometry, error functions,
activations, integral rounding, and binary exponent operations preserve gradual
underflow.
Both scalar and vector constant folding and the Core VM call these same typed
functions:

```c
float root = iree_math_sqrt_f32(x);
double weight = iree_math_exp_f64(score);
double power = iree_math_exp2_f64(exponent);
double correction = iree_math_log1p_f64(delta);
```

Integral rounding provides ceil, floor, truncation, nearest-even, and
nearest-away at both widths. Fixed-direction native instructions or integer
exponent/mantissa rounding preserve signed zero and canonicalize NaNs without
depending on the caller's rounding direction or platform libm.

`frexp` decomposes both widths exactly using integer normalization, including
subnormal inputs. `ldexp` accepts every signed i32 exponent and uses bounded
power-of-two multiplications with a single final rounding. Software FMA shares
that scaling implementation.

Square root is correctly rounded and uses hardware instructions when available.
An integer restoring algorithm supplies the same payloads on hardware without
square root. Reciprocal square root rounds after both square root and division.
The exponentials use Arm's table-and-polynomial algorithms with fixed unfused
rounding points and at most one adjacent value of error from correct rounding.
`expm1` uses compensated reduction and a rational approximation to retain
small differences that would vanish in `exp(x)-1`; both widths share that
algorithm, with parallel polynomial evaluation for F64.
Their exceptional-value behavior and arithmetic profile are part of their
stable mapping, not host library policy.

Cube root uses a bit estimate followed by fixed Newton refinement. F32 uses
two F64 refinements; F64 uses a short reciprocal-cube-root polynomial and one
final correction. Both preserve signed zero and infinity, accept negative
inputs, and normalize subnormal inputs without flushing them.

Natural and base-two logarithms use table reduction and short polynomials.
The F32 natural and base-ten functions share a F64 kernel until the final
rounding. F64 logarithms retain split reduction centers to bound rounding error
without depending on FMA availability. `log1p` preserves the correction lost
when forming `1+x`, including inputs so small that the sum rounds to one.

Real powers retain extra logarithm precision through exponent multiplication:
F32 evaluates its logarithm in F64, and F64 carries a high part and tail. They
share exponential reconstruction, including signed subnormal results, while
handling negative bases and exponent parity explicitly. Composing ordinary
rounded `log` and `exp` calls would discard the precision needed by `pow`.

Hyperbolic functions share the owned exponentials, logarithms, and roots.
Cancellation-safe interval formulas preserve small sinh/asinh/atanh results and
acosh near one. Scaled exponential reconstruction retains finite sinh/cosh
results even when exp would overflow. Compact rational tanh approximations
avoid cancellation without adding another exponential table.

Inverse trigonometry shares rational asin/acos corrections and compensated
root reconstruction near the endpoints. atan reduces into four polynomial
intervals; atan2 reuses it after guarding the ratio exponent and retains signed
axes and quadrants explicitly. These are ordinary radian-valued mappings.

Error functions share rational tail reduction and a compensated exponential.
`erfc` retains its small positive tail directly, including subnormal results.
A compact interval polynomial supplies the cancellation-sensitive F64 central
region shared by `erf` and `erfc`; F32 uses its own shorter rational coefficients
with widened intermediates where complement reconstruction requires them.

Activations share those numerical leaves. Sign-split sigmoid and softplus
avoid overflow and cancellation; SiLU and the three authored GELU variants
retain their scaled negative tails before final underflow. Erf GELU reuses the
complementary error-function tail, while tanh and scaled-logistic GELU retain
their distinct formulas and scale semantics. These composed functions have
condition-dependent error bounds documented in `activation.h`.

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
The ordinary mappings require gradual underflow, while approximation leaves
own their documented flush rules.

## Organization

Operations are grouped by algorithm family and provenance:

- `exponential.h` owns natural/base-two exponentials, `expm1`, and real powers.
- `logarithm.h` owns natural/base-two/base-ten logarithms and `log1p`.
- `hyperbolic.h` owns sinh/cosh/tanh and their inverses.
- `inverse_trigonometry.h` owns asin/acos/atan/atan2.
- `error_functions.h` owns erf and erfc.
- `activation.h` owns sigmoid, SiLU, softplus, and the three GELU variants.
- `fma.h` owns fused arithmetic and narrow-format intermediate evaluation.
- `rounding.h` owns the five integral rounding directions.
- `scaling.h` owns binary exponent decomposition and scaling.
- `roots.h` owns reciprocal, reciprocal square root, square root, and cube root.
- `trigonometry.h` owns radian sine/cosine/tangent and turns-based sine/cosine.

All clusters form one `:math` build target. Keeping their implementations in
separate translation units preserves readable ownership and ordinary
dead-stripping without multiplying build graph actions. `float_bits.h` is a
private representation helper, not another public API.

New operations require a concrete compiler or runtime consumer, an explicit
payload contract, independent accuracy evidence, cross-platform bit vectors,
and measured latency and closure size. Target-specific approximations belong
with their target unless they are selected as the portable IREE mapping.
