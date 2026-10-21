// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/cast.h"

#include "iree/base/internal/math.h"

static bool loom_index_cast_to_offset_zero_extends_source(
    loom_scalar_type_t input_type, loom_scalar_type_t result_type,
    int32_t input_bitwidth) {
  return result_type == LOOM_SCALAR_TYPE_OFFSET &&
         loom_scalar_type_is_integer(input_type) && input_bitwidth < 63;
}

static loom_value_facts_t loom_index_cast_zero_extend_to_offset_facts(
    loom_value_facts_t facts, int32_t input_bitwidth) {
  const int64_t unsigned_extent = INT64_C(1) << input_bitwidth;
  const int64_t unsigned_max = unsigned_extent - 1;
  if (facts.range_lo >= 0) {
    return loom_value_facts_clamp_domain(facts, 0, unsigned_max);
  }
  if (facts.range_hi < 0) {
    return loom_value_facts_make(
        facts.range_lo + unsigned_extent, facts.range_hi + unsigned_extent,
        iree_math_gcd_i64(facts.known_divisor, unsigned_extent));
  }
  return loom_value_facts_make(0, unsigned_max, 1);
}

loom_value_facts_t loom_index_cast_transfer_facts(
    loom_scalar_type_t input_type, loom_scalar_type_t result_type,
    loom_value_facts_t input_facts) {
  int64_t input_lo = 0;
  int64_t input_hi = 0;
  int64_t result_lo = 0;
  int64_t result_hi = 0;
  if (!loom_value_facts_scalar_type_domain(input_type, &input_lo, &input_hi) ||
      !loom_value_facts_scalar_type_domain(result_type, &result_lo,
                                           &result_hi)) {
    return loom_value_facts_unknown();
  }

  const int32_t input_bitwidth = loom_scalar_type_bitwidth(input_type);
  loom_value_facts_t facts =
      loom_value_facts_clamp_domain(input_facts, input_lo, input_hi);
  if (loom_index_cast_to_offset_zero_extends_source(input_type, result_type,
                                                    input_bitwidth)) {
    return loom_index_cast_zero_extend_to_offset_facts(facts, input_bitwidth);
  }

  if (facts.range_lo >= result_lo && facts.range_hi <= result_hi) {
    return facts;
  }

  // Truncation can wrap, and a signed-to-offset cast requires an input proof.
  // Intersecting an unproven input with the result domain could manufacture an
  // exact value and let folding erase the cast before legality checks it.
  return loom_value_facts_make(result_lo, result_hi, 1);
}

bool loom_index_cast_roundtrip_preserves_value(
    loom_scalar_type_t address_type, loom_scalar_type_t payload_type,
    loom_value_facts_t source_facts) {
  if ((address_type != LOOM_SCALAR_TYPE_INDEX &&
       address_type != LOOM_SCALAR_TYPE_OFFSET) ||
      !loom_scalar_type_is_integer(payload_type)) {
    return false;
  }

  const int32_t payload_bit_count = loom_scalar_type_bitwidth(payload_type);
  if (payload_bit_count <= 0 || payload_bit_count > 64) {
    return false;
  }
  const bool uses_unsigned_domain = payload_type == LOOM_SCALAR_TYPE_I1 ||
                                    address_type == LOOM_SCALAR_TYPE_OFFSET;
  return uses_unsigned_domain ? loom_value_facts_fit_unsigned_bit_count(
                                    source_facts, (uint8_t)payload_bit_count)
                              : loom_value_facts_fit_signed_bit_count(
                                    source_facts, (uint8_t)payload_bit_count);
}
