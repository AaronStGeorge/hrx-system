// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent contraction request records.
//
// Contract requests preserve algebra, operand roles, payload types, fragment
// summaries, and lowering policy before a target adapter projects the request
// into a target-owned descriptor table. The generic layer must not name target
// descriptors, opcodes, native packets, or LLVM intrinsics.

#ifndef LOOM_ANALYSIS_CONTRACT_H_
#define LOOM_ANALYSIS_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/analysis/policy.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_contract_kind_e {
  // Unknown or uninitialized contract kind.
  LOOM_CONTRACT_KIND_UNKNOWN = 0,
  // Matrix multiply-like M/N/K contraction.
  LOOM_CONTRACT_KIND_MATRIX_MULTIPLY = 1,
  // Vector dot-product contraction.
  LOOM_CONTRACT_KIND_VECTOR_DOT = 2,
  // Outer-product contraction.
  LOOM_CONTRACT_KIND_OUTER_PRODUCT = 3,
} loom_contract_kind_t;

typedef enum loom_contract_arithmetic_e {
  // Unknown or uninitialized arithmetic family.
  LOOM_CONTRACT_ARITHMETIC_UNKNOWN = 0,
  // Floating-point multiply-add style contraction.
  LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT = 1,
  // Integer multiply-add style contraction.
  LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT = 2,
  // Mixed payload/accumulator contraction.
  LOOM_CONTRACT_ARITHMETIC_MIXED_DOT = 3,
  // Contraction depends on an explicit numeric transform.
  LOOM_CONTRACT_ARITHMETIC_TRANSFORM_BACKED = 4,
} loom_contract_arithmetic_t;

typedef enum loom_contract_operand_role_e {
  // Unknown or uninitialized operand role.
  LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN = 0,
  // Left-hand source operand.
  LOOM_CONTRACT_OPERAND_ROLE_LHS = 1,
  // Right-hand source operand.
  LOOM_CONTRACT_OPERAND_ROLE_RHS = 2,
  // Accumulator input operand.
  LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR = 3,
  // Result operand.
  LOOM_CONTRACT_OPERAND_ROLE_RESULT = 4,
} loom_contract_operand_role_t;

typedef enum loom_contract_numeric_type_e {
  // Unknown or uninitialized numeric payload.
  LOOM_CONTRACT_NUMERIC_UNKNOWN = 0,
  // The payload is interpreted as signed i4 elements.
  LOOM_CONTRACT_NUMERIC_I4 = 1,
  // The payload is interpreted as unsigned i4 elements.
  LOOM_CONTRACT_NUMERIC_U4 = 2,
  // The payload is interpreted as signed i8 elements.
  LOOM_CONTRACT_NUMERIC_I8 = 3,
  // The payload is interpreted as unsigned i8 elements.
  LOOM_CONTRACT_NUMERIC_U8 = 4,
  // The payload is interpreted as signed i16 elements.
  LOOM_CONTRACT_NUMERIC_I16 = 5,
  // The payload is interpreted as unsigned i16 elements.
  LOOM_CONTRACT_NUMERIC_U16 = 6,
  // The payload is interpreted as signed i32 elements.
  LOOM_CONTRACT_NUMERIC_I32 = 7,
  // The payload is interpreted as unsigned i32 elements.
  LOOM_CONTRACT_NUMERIC_U32 = 8,
  // The payload is interpreted as IEEE f16 elements.
  LOOM_CONTRACT_NUMERIC_F16 = 9,
  // The payload is interpreted as BF16 elements.
  LOOM_CONTRACT_NUMERIC_BF16 = 10,
  // The payload is interpreted as IEEE f32 elements.
  LOOM_CONTRACT_NUMERIC_F32 = 11,
  // The payload is interpreted as IEEE f64 elements.
  LOOM_CONTRACT_NUMERIC_F64 = 12,
  // The payload is interpreted as FP8 elements.
  LOOM_CONTRACT_NUMERIC_FP8 = 13,
  // The payload is interpreted as BF8 elements.
  LOOM_CONTRACT_NUMERIC_BF8 = 14,
  // The payload is interpreted as FP6 elements.
  LOOM_CONTRACT_NUMERIC_FP6 = 15,
  // The payload is interpreted as BF6 elements.
  LOOM_CONTRACT_NUMERIC_BF6 = 16,
  // The payload is interpreted as FP4 elements.
  LOOM_CONTRACT_NUMERIC_FP4 = 17,
} loom_contract_numeric_type_t;

typedef enum loom_contract_scale_kind_e {
  // Unknown or uninitialized scale requirement.
  LOOM_CONTRACT_SCALE_UNKNOWN = 0,
  // No explicit scale operands are required.
  LOOM_CONTRACT_SCALE_NONE = 1,
  // 32-bit scale exponent operands are required.
  LOOM_CONTRACT_SCALE_32 = 2,
  // 16-bit scale exponent operands are required.
  LOOM_CONTRACT_SCALE_16 = 3,
} loom_contract_scale_kind_t;

typedef enum loom_contract_capability_class_e {
  // Unknown or no target primitive capability class.
  LOOM_CONTRACT_CAPABILITY_CLASS_UNKNOWN = 0,
  // CPU packed-dot/vector-dot target primitive class.
  LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT = 1,
  // GPU matrix contract target primitive class.
  LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX = 2,
  // Target tensor-memory movement class.
  LOOM_CONTRACT_CAPABILITY_CLASS_TENSOR_MOVEMENT = 3,
  // Target or runtime ukernel contract class.
  LOOM_CONTRACT_CAPABILITY_CLASS_UKERNEL = 4,
} loom_contract_capability_class_t;

typedef uint32_t loom_contract_fragment_atom_bits_t;

// The fragment carries a logical algebra axis.
#define LOOM_CONTRACT_FRAGMENT_LOGICAL ((uint32_t)1u << 0)
// The fragment carries a physical storage-record axis.
#define LOOM_CONTRACT_FRAGMENT_STORAGE_RECORD ((uint32_t)1u << 1)
// The fragment carries payload-internal grouping.
#define LOOM_CONTRACT_FRAGMENT_INTERNAL ((uint32_t)1u << 2)
// The fragment carries CPU/vector lane ownership.
#define LOOM_CONTRACT_FRAGMENT_VECTOR_LANE ((uint32_t)1u << 3)
// The fragment groups multiple target instructions for one logical contract.
#define LOOM_CONTRACT_FRAGMENT_INSTRUCTION_GROUP ((uint32_t)1u << 4)
// The fragment carries subgroup or wave lane ownership.
#define LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE ((uint32_t)1u << 5)
// The fragment carries workgroup-level ownership.
#define LOOM_CONTRACT_FRAGMENT_WORKGROUP ((uint32_t)1u << 6)
// The fragment carries bank or swizzle-sensitive ownership.
#define LOOM_CONTRACT_FRAGMENT_MEMORY_BANK ((uint32_t)1u << 7)

typedef uint32_t loom_contract_capability_flags_t;

// The request has sparse metadata operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_SPARSE_METADATA ((uint32_t)1u << 0)
// The request has explicit scale operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS ((uint32_t)1u << 1)
// The request has matrix or payload format-selector operands or facts
// available.
#define LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS ((uint32_t)1u << 2)
// The request has target reuse operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_REUSE ((uint32_t)1u << 3)
// The request has clamp operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_CLAMP ((uint32_t)1u << 4)
// The request has sign-selection operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_SIGN_SELECT ((uint32_t)1u << 5)
// The request has operand modifier operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_OPERAND_MODIFIERS ((uint32_t)1u << 6)
// The request has accumulator modifier operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_ACCUMULATOR_MODIFIER ((uint32_t)1u << 7)
// The request has op-select operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_OPSEL ((uint32_t)1u << 8)
// The request has scale-format selector operands or facts available.
#define LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS ((uint32_t)1u << 9)
// The request permits zero-scale fallback to an unscaled primitive.
#define LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK ((uint32_t)1u << 10)

typedef struct loom_contract_shape_t {
  // Exact M/result-row extent, or 0 when unknown.
  int64_t m;

  // Exact N/result-column extent, or 0 when unknown.
  int64_t n;

  // Exact K/reduction extent, or 0 when unknown.
  int64_t k;
} loom_contract_shape_t;

typedef struct loom_contract_operand_t {
  // Role this operand plays in the contraction.
  loom_contract_operand_role_t role;

  // Logical numeric interpretation of this operand payload.
  loom_contract_numeric_type_t numeric_type;

  // Target-fragment payload register count when already known, or 0.
  uint16_t payload_register_count;

  // Logical scalar element count represented by the payload, or 0.
  uint16_t payload_element_count;
} loom_contract_operand_t;

typedef struct loom_contract_fragment_t {
  // Bitset of loom_contract_fragment_atom_bits_t values.
  loom_contract_fragment_atom_bits_t atom_bits;

  // Native vector width in bits for CPU/vector fragments, or 0.
  uint16_t vector_bit_width;

  // Source lanes consumed from each input vector, or 0.
  uint16_t source_lane_count;

  // Accumulator/result lanes produced by one vector primitive, or 0.
  uint16_t result_lane_count;

  // Subgroup or wave size for GPU fragments, or 0 when target-owned.
  uint16_t subgroup_size;
} loom_contract_fragment_t;

typedef struct loom_contract_request_t {
  // High-level contraction kind.
  loom_contract_kind_t kind;

  // Arithmetic family for the contraction.
  loom_contract_arithmetic_t arithmetic;

  // Exact logical matrix/vector contract shape.
  loom_contract_shape_t shape;

  // Number of K payload elements reduced into each accumulator contribution.
  uint16_t k_group_size;

  // Left-hand source operand facts.
  loom_contract_operand_t lhs;

  // Right-hand source operand facts.
  loom_contract_operand_t rhs;

  // Accumulator operand facts.
  loom_contract_operand_t accumulator;

  // Result operand facts.
  loom_contract_operand_t result;

  // Fragment ownership facts required to preserve target-shaped structure.
  loom_contract_fragment_t fragment;

  // Requested target primitive capability class.
  loom_contract_capability_class_t capability_class;

  // Bitset of capability facts or operands available to target adapters.
  loom_contract_capability_flags_t available_capability_flags;

  // Bitset of capability facts the selected target primitive must require.
  loom_contract_capability_flags_t required_capability_flags;

  // Explicit scale operand kind required by this contract.
  loom_contract_scale_kind_t scale_kind;

  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_contract_request_t;

typedef uint32_t loom_contract_rejection_bits_t;

// No generic rejection reason was recorded.
#define LOOM_CONTRACT_REJECTION_NONE ((uint32_t)0u)
// The request itself was invalid or absent.
#define LOOM_CONTRACT_REJECTION_INVALID_REQUEST ((uint32_t)1u << 0)
// The exact M/N/K shape or K grouping was missing or invalid.
#define LOOM_CONTRACT_REJECTION_SHAPE ((uint32_t)1u << 1)
// One or more operand roles were missing or inconsistent.
#define LOOM_CONTRACT_REJECTION_ROLE ((uint32_t)1u << 2)
// One or more numeric payload facts were missing or unsupported.
#define LOOM_CONTRACT_REJECTION_NUMERIC ((uint32_t)1u << 3)
// Fragment facts required by the requested capability class were missing.
#define LOOM_CONTRACT_REJECTION_FRAGMENT ((uint32_t)1u << 4)
// The requested capability class or capability flags were unsupported.
#define LOOM_CONTRACT_REJECTION_CAPABILITY ((uint32_t)1u << 5)
// Lowering policy does not permit the only available lowering family.
#define LOOM_CONTRACT_REJECTION_POLICY ((uint32_t)1u << 6)

typedef struct loom_contract_diagnostic_t {
  // Bitset of loom_contract_rejection_bits_t values.
  loom_contract_rejection_bits_t rejection_bits;
} loom_contract_diagnostic_t;

// Initializes a request with explicit unknown values and reference-allowed
// policy. Callers then fill the fields they have proven.
void loom_contract_request_initialize(loom_contract_request_t* out_request);

// Validates the target-independent fields required before target projection.
bool loom_contract_request_validate(const loom_contract_request_t* request,
                                    loom_contract_diagnostic_t* out_diagnostic);

// Maps a Loom scalar type to a generic contract numeric type.
bool loom_contract_numeric_type_from_scalar(
    loom_scalar_type_t scalar_type, bool unsigned_integer,
    loom_contract_numeric_type_t* out_numeric_type);

// Returns the stable diagnostic spelling for a generic numeric type.
iree_string_view_t loom_contract_numeric_type_name(
    loom_contract_numeric_type_t numeric_type);

// Returns a diagnostic detail string for generic contract rejection flags.
iree_string_view_t loom_contract_rejection_detail(
    loom_contract_rejection_bits_t rejection_bits);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_H_
