// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Value facts: per-value dataflow analysis properties.
//
// A loom_value_facts_t is a 32-byte summary of what is known about an
// SSA value: signed integer range [lo, hi], largest known divisor, and
// cached predicate flags (non-negative, non-zero, power-of-two, etc.).
//
// Facts are the building blocks of loom's always-on value analysis.
// They are maintained incrementally by the rewriter during
// canonicalization and propagated through arithmetic via transfer
// functions. Constants produce exact facts (lo == hi). Assume
// predicates tighten facts. Arithmetic transfer functions compute
// output facts from input facts.
//
// Constructors and predicates are inline (called in tight pattern
// matching loops). Transfer functions and predicate application are
// out-of-line in facts.c (called once per op during propagation).
//
// Transfer functions take facts by pointer and write the result into
// an output parameter. The output MAY alias one input for in-place
// accumulation (e.g., computing element counts by chaining muli):
//
//   loom_value_facts_t accumulator = loom_value_facts_exact_i64(1);
//   for (uint8_t i = 0; i < rank; ++i) {
//     loom_value_facts_t dim_facts = facts[dim_value_id];
//     loom_value_facts_muli(&accumulator, &dim_facts, &accumulator);
//   }

#ifndef LOOM_IR_FACTS_H_
#define LOOM_IR_FACTS_H_

#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/math.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Flags
//===----------------------------------------------------------------------===//

// Cached predicate flags derived from range and divisor. These are
// always consistent with the range/divisor fields — constructors
// compute them, and they are never written independently.
enum loom_value_fact_flag_bits_e {
  // The signed range lower bound is >= 0.
  LOOM_VALUE_FACT_NON_NEGATIVE = 1u << 0,
  // The range does not include zero (lo > 0 or hi < 0).
  LOOM_VALUE_FACT_NON_ZERO = 1u << 1,
  // The signed range lower bound is > 0. Implies NON_NEGATIVE and
  // NON_ZERO.
  LOOM_VALUE_FACT_POSITIVE = 1u << 2,
  // The value is known to be a power of two. Set by pow2() predicates
  // or when an exact value is a power of two. Preserved across
  // recompute_flags since it may come from predicates rather than
  // range analysis.
  LOOM_VALUE_FACT_POWER_OF_TWO = 1u << 3,
  // The range is a single point (range_lo == range_hi). The value is
  // a known compile-time constant. For integer types, range_lo is the
  // constant value. For float types, range_lo contains the IEEE 754
  // bit pattern of the double (see FLOAT flag).
  LOOM_VALUE_FACT_EXACT = 1u << 4,
  // The range is [0, 1]. Typical for i1 comparison results.
  LOOM_VALUE_FACT_BOOLEAN = 1u << 5,
  // The value has a floating-point type. When EXACT is also set,
  // range_lo/range_hi contain the IEEE 754 double bit pattern (via
  // memcpy), not an integer range. Without EXACT, float facts are
  // unknown (no float range analysis).
  LOOM_VALUE_FACT_FLOAT = 1u << 6,
};
typedef uint32_t loom_value_fact_flags_t;

// Context-local extension payload ID. Zero means the fact has no extension.
typedef uint32_t loom_value_fact_extension_id_t;
#define LOOM_VALUE_FACT_EXTENSION_ID_NONE ((loom_value_fact_extension_id_t)0)

//===----------------------------------------------------------------------===//
// Struct
//===----------------------------------------------------------------------===//

// Per-value analysis facts. 32 bytes, cache-friendly for dense arrays.
typedef struct loom_value_facts_t {
  // Signed integer range [lo, hi], inclusive. Default (unknown):
  // [INT64_MIN, INT64_MAX]. For float types with EXACT flag set,
  // range_lo contains the IEEE 754 bit pattern of the double value
  // (via memcpy); range_hi == range_lo.
  int64_t range_lo;
  int64_t range_hi;

  // Largest known divisor (>= 1). GCD semantics: if the value is
  // known to be a multiple of both 16 and 24, known_divisor =
  // gcd(16, 24) = 8. For exact integer values, known_divisor = |value|
  // (or 1 if value is 0 or INT64_MIN). Default (unknown): 1.
  int64_t known_divisor;

  // Cached predicate bitflags. Derived from range/divisor but cheaper
  // to test (single bit check vs. integer comparison).
  uint32_t flags;

  // One-based ID into the current fact context extension table. Zero means no
  // extension. Extension IDs are context-local and must only be interpreted by
  // APIs that receive the context that produced the facts.
  loom_value_fact_extension_id_t extension_id;
} loom_value_facts_t;

static_assert(sizeof(loom_value_facts_t) == 32,
              "loom_value_facts_t must be 32 bytes");

//===----------------------------------------------------------------------===//
// Constructors (inline)
//===----------------------------------------------------------------------===//

// Unknown facts: the conservative default. No information about the
// value beyond its type.
static inline loom_value_facts_t loom_value_facts_unknown(void) {
  loom_value_facts_t facts = {0};
  facts.range_lo = INT64_MIN;
  facts.range_hi = INT64_MAX;
  facts.known_divisor = 1;
  return facts;
}

// Exact integer value. Computes all flags from the value.
loom_value_facts_t loom_value_facts_exact_i64(int64_t value);

// Exact float value. Stores the IEEE 754 bit pattern in range_lo.
// Float facts are exact-only (no float range analysis).
loom_value_facts_t loom_value_facts_exact_f64(double value);

// Extracts the double from an exact float fact.
static inline double loom_value_facts_as_f64(loom_value_facts_t facts) {
  double value;
  memcpy(&value, &facts.range_lo, sizeof(double));
  return value;
}

// General constructor: range with divisor. Computes flags from the
// range and divisor values.
loom_value_facts_t loom_value_facts_make(int64_t lo, int64_t hi,
                                         int64_t known_divisor);

//===----------------------------------------------------------------------===//
// Predicates (inline)
//===----------------------------------------------------------------------===//

static inline bool loom_value_facts_is_exact(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_EXACT) != 0;
}

static inline bool loom_value_facts_is_non_negative(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_NON_NEGATIVE) != 0;
}

static inline bool loom_value_facts_is_non_zero(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_NON_ZERO) != 0;
}

static inline bool loom_value_facts_is_positive(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_POSITIVE) != 0;
}

static inline bool loom_value_facts_is_power_of_two(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_POWER_OF_TWO) != 0;
}

static inline bool loom_value_facts_is_boolean(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_BOOLEAN) != 0;
}

static inline bool loom_value_facts_is_float(loom_value_facts_t facts) {
  return (facts.flags & LOOM_VALUE_FACT_FLOAT) != 0;
}

static inline bool loom_value_facts_divisible_by(loom_value_facts_t facts,
                                                 int64_t divisor) {
  return facts.known_divisor % divisor == 0;
}

// Returns true if the facts carry no information (full range, unit
// divisor, no flags).
static inline bool loom_value_facts_is_unknown(loom_value_facts_t facts) {
  return facts.range_lo == INT64_MIN && facts.range_hi == INT64_MAX &&
         facts.known_divisor == 1 && facts.flags == 0 &&
         facts.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE;
}

//===----------------------------------------------------------------------===//
// Comparison (inline)
//===----------------------------------------------------------------------===//

// Equality check for propagation damping. 32-byte memcmp.
static inline bool loom_value_facts_equal(loom_value_facts_t a,
                                          loom_value_facts_t b) {
  return memcmp(&a, &b, sizeof(loom_value_facts_t)) == 0;
}

//===----------------------------------------------------------------------===//
// Meet (inline)
//===----------------------------------------------------------------------===//

// Conservative join: widens range to the outer bounds, weakens divisor
// to the GCD. Used at join points (scf.if yields, block arguments).
static inline void loom_value_facts_meet(
    const loom_value_facts_t* IREE_RESTRICT a,
    const loom_value_facts_t* IREE_RESTRICT b,
    loom_value_facts_t* IREE_RESTRICT out) {
  *out = loom_value_facts_make(
      a->range_lo < b->range_lo ? a->range_lo : b->range_lo,
      a->range_hi > b->range_hi ? a->range_hi : b->range_hi,
      loom_gcd_i64(a->known_divisor, b->known_divisor));
}

//===----------------------------------------------------------------------===//
// Predicate application (defined in facts.c)
//===----------------------------------------------------------------------===//

// Recomputes the cached flags from range_lo, range_hi, and
// known_divisor. Preserves POWER_OF_TWO and FLOAT flags if already
// set (they may come from predicates, not just range analysis).
void loom_value_facts_recompute_flags(loom_value_facts_t* facts);

// Tightens facts using a single predicate constraint. Modifies the
// facts in place and recomputes flags afterward.
void loom_value_facts_apply_predicate(loom_value_facts_t* facts,
                                      const loom_predicate_t* predicate);

//===----------------------------------------------------------------------===//
// Transfer functions (defined in facts.c)
//===----------------------------------------------------------------------===//
//
// Compute output facts from input facts for each arithmetic operation.
// The output pointer MAY alias one input for in-place accumulation
// (e.g., loom_value_facts_muli(&acc, &dim, &acc)). Do not alias both
// inputs with the output simultaneously.
//
// For exact inputs, these produce exact outputs (constant folding).
// For range inputs, they compute range bounds with checked arithmetic
// and fall back to unknown on overflow.

// Binary arithmetic.
void loom_value_facts_addi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_subi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_muli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_divui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_divsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_remui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_remsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);

// Shifts.
void loom_value_facts_shli(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_shrui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_shrsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);

// Bitwise.
void loom_value_facts_andi(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);
void loom_value_facts_ori(const loom_value_facts_t* lhs,
                          const loom_value_facts_t* rhs,
                          loom_value_facts_t* out);
void loom_value_facts_xori(const loom_value_facts_t* lhs,
                           const loom_value_facts_t* rhs,
                           loom_value_facts_t* out);

// Min / max.
void loom_value_facts_minsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_maxsi(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_minui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);
void loom_value_facts_maxui(const loom_value_facts_t* lhs,
                            const loom_value_facts_t* rhs,
                            loom_value_facts_t* out);

// Unary.
void loom_value_facts_negi(const loom_value_facts_t* input,
                           loom_value_facts_t* out);
void loom_value_facts_absi(const loom_value_facts_t* input,
                           loom_value_facts_t* out);

// Ternary: fused multiply-add (a*b + c).
void loom_value_facts_fmai(const loom_value_facts_t* a,
                           const loom_value_facts_t* b,
                           const loom_value_facts_t* c,
                           loom_value_facts_t* out);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_FACTS_H_
