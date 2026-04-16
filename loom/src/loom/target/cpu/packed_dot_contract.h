// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CPU packed dot target contracts.
//
// These descriptors model target-native vector dot instructions such as x86
// VNNI and BF16 dot-product instructions. They are intentionally narrower than
// matrix tile contracts: the contract consumes packed register vectors, carries
// exact lane signedness from the consuming op, and leaves memory packing/layout
// facts to the view/encoding substrate.

#ifndef LOOM_TARGET_CPU_PACKED_DOT_CONTRACT_H_
#define LOOM_TARGET_CPU_PACKED_DOT_CONTRACT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bitset of CPU packed-dot features available on a selected target.
typedef uint64_t loom_cpu_packed_dot_feature_bits_t;

// Target supports AVX-512 VNNI byte/word dot products.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VNNI (UINT64_C(1) << 0)
// Target supports AVX-512 VL encodings for 128-bit and 256-bit forms.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_VL (UINT64_C(1) << 1)
// Target supports VEX AVX-VNNI 128-bit and 256-bit forms.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI (UINT64_C(1) << 2)
// Target supports AVX-VNNI-INT8 byte dot-product forms.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI_INT8 (UINT64_C(1) << 3)
// Target supports AVX-VNNI-INT16 word dot-product forms.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX_VNNI_INT16 (UINT64_C(1) << 4)
// Target supports AVX10.2 extended VNNI dot products.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX10_2 (UINT64_C(1) << 5)
// Target supports AVX-512 BF16 dot products.
#define LOOM_CPU_PACKED_DOT_FEATURE_X86_AVX512_BF16 (UINT64_C(1) << 6)

typedef enum loom_cpu_packed_dot_family_e {
  // Unknown or uninitialized packed-dot family.
  LOOM_CPU_PACKED_DOT_FAMILY_UNKNOWN = 0,
  // x86 AVX-512 VNNI instruction family.
  LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_VNNI,
  // x86 AVX-VNNI instruction family.
  LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI,
  // x86 AVX-VNNI-INT8 instruction family.
  LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT8,
  // x86 AVX-VNNI-INT16 instruction family.
  LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX_VNNI_INT16,
  // x86 AVX10.2 VNNI instruction family.
  LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX10_2,
  // x86 AVX-512 BF16 dot-product instruction family.
  LOOM_CPU_PACKED_DOT_FAMILY_X86_AVX512_BF16,
} loom_cpu_packed_dot_family_t;

typedef enum loom_cpu_packed_dot_numeric_type_e {
  // Unknown or uninitialized numeric payload.
  LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN = 0,
  // Signed i8 payload lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_I8,
  // Unsigned i8 payload lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_U8,
  // Signed i16 payload lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_I16,
  // Unsigned i16 payload lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_U16,
  // IEEE f16 payload lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_F16,
  // BF16 payload lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_BF16,
  // Signed i32 accumulator/result lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_I32,
  // IEEE f32 accumulator/result lanes.
  LOOM_CPU_PACKED_DOT_NUMERIC_F32,
} loom_cpu_packed_dot_numeric_type_t;

typedef enum loom_cpu_packed_dot_llvm_source_abi_e {
  // Unknown or uninitialized LLVM intrinsic source operand ABI.
  LOOM_CPU_PACKED_DOT_LLVM_SOURCE_ABI_UNKNOWN = 0,
  // LLVM intrinsic source operands use the logical payload vector type.
  LOOM_CPU_PACKED_DOT_LLVM_SOURCE_ABI_PAYLOAD,
  // LLVM intrinsic source operands use the accumulator register vector type.
  LOOM_CPU_PACKED_DOT_LLVM_SOURCE_ABI_ACCUMULATOR_VECTOR,
} loom_cpu_packed_dot_llvm_source_abi_t;

// Bitset of semantic flags required by a packed-dot descriptor.
typedef uint32_t loom_cpu_packed_dot_contract_flags_t;

// Descriptor performs a saturating add into the accumulator lane.
#define LOOM_CPU_PACKED_DOT_CONTRACT_FLAG_SATURATING ((uint32_t)1u << 0)

typedef struct loom_cpu_packed_dot_shape_t {
  // Native vector width in bits.
  uint16_t vector_bit_width;
  // Number of source lanes consumed from each input vector.
  uint16_t input_lane_count;
  // Number of accumulator/result lanes produced by the vector instruction.
  uint16_t result_lane_count;
  // Number of source lane products reduced into each accumulator lane.
  uint16_t reduction_group_size;
} loom_cpu_packed_dot_shape_t;

typedef struct loom_cpu_packed_dot_descriptor_t {
  // Stable Loom descriptor name used by tests, diagnostics, and target logs.
  iree_string_view_t name;
  // LLVM intrinsic name when Loom lowers through an intrinsic call.
  iree_string_view_t llvm_intrinsic_name;
  // Source operand ABI required by the LLVM intrinsic declaration and call.
  loom_cpu_packed_dot_llvm_source_abi_t llvm_source_abi;
  // Assembly mnemonic expected after LLVM instruction selection.
  iree_string_view_t instruction_mnemonic;
  // CPU instruction family used by this descriptor.
  loom_cpu_packed_dot_family_t family;
  // Target feature bits required before this descriptor is legal.
  loom_cpu_packed_dot_feature_bits_t required_feature_bits;
  // Optional semantic flags carried by this descriptor.
  loom_cpu_packed_dot_contract_flags_t flags;
  // Register lane shape consumed and produced by one vector instruction.
  loom_cpu_packed_dot_shape_t shape;
  // Left-hand source payload numeric interpretation.
  loom_cpu_packed_dot_numeric_type_t lhs_numeric_type;
  // Right-hand source payload numeric interpretation.
  loom_cpu_packed_dot_numeric_type_t rhs_numeric_type;
  // Accumulator lane numeric type.
  loom_cpu_packed_dot_numeric_type_t accumulator_numeric_type;
  // Result lane numeric type.
  loom_cpu_packed_dot_numeric_type_t result_numeric_type;
} loom_cpu_packed_dot_descriptor_t;

// Bitset of structural reasons that prevented a packed-dot contract match.
typedef uint32_t loom_cpu_packed_dot_rejection_bits_t;

// No rejection reason was recorded.
#define LOOM_CPU_PACKED_DOT_REJECTION_NONE ((uint32_t)0u)
// The requested family rejected every descriptor.
#define LOOM_CPU_PACKED_DOT_REJECTION_FAMILY ((uint32_t)1u << 0)
// The requested vector/register shape rejected every family-compatible
// descriptor.
#define LOOM_CPU_PACKED_DOT_REJECTION_SHAPE ((uint32_t)1u << 1)
// The requested payload numeric types rejected every shape-compatible
// descriptor.
#define LOOM_CPU_PACKED_DOT_REJECTION_PAYLOAD ((uint32_t)1u << 2)
// The request required flags unsupported by the remaining candidates.
#define LOOM_CPU_PACKED_DOT_REJECTION_FLAGS ((uint32_t)1u << 3)
// The selected CPU feature bits rejected every semantic candidate.
#define LOOM_CPU_PACKED_DOT_REJECTION_FEATURES ((uint32_t)1u << 4)
// The request itself was invalid or absent.
#define LOOM_CPU_PACKED_DOT_REJECTION_INVALID_REQUEST ((uint32_t)1u << 5)

typedef struct loom_cpu_packed_dot_match_request_t {
  // Optional instruction family. UNKNOWN allows any family.
  loom_cpu_packed_dot_family_t family;
  // Required register lane shape.
  loom_cpu_packed_dot_shape_t shape;
  // Required left-hand source numeric interpretation.
  loom_cpu_packed_dot_numeric_type_t lhs_numeric_type;
  // Required right-hand source numeric interpretation.
  loom_cpu_packed_dot_numeric_type_t rhs_numeric_type;
  // Required accumulator lane numeric type.
  loom_cpu_packed_dot_numeric_type_t accumulator_numeric_type;
  // Required result lane numeric type.
  loom_cpu_packed_dot_numeric_type_t result_numeric_type;
  // Processor feature bits available to the target.
  loom_cpu_packed_dot_feature_bits_t feature_bits;
  // Descriptor flags the selected contract must carry.
  loom_cpu_packed_dot_contract_flags_t required_flags;
} loom_cpu_packed_dot_match_request_t;

typedef struct loom_cpu_packed_dot_match_diagnostic_t {
  // Structural rejection reason selected for user-facing diagnostics.
  loom_cpu_packed_dot_rejection_bits_t rejection_bits;
  // Number of descriptors scanned.
  iree_host_size_t descriptor_count;
  // Number of descriptors that matched the requested family.
  iree_host_size_t family_candidate_count;
  // Number of family-compatible descriptors that matched the requested shape.
  iree_host_size_t shape_candidate_count;
  // Number of shape-compatible descriptors that matched payload numeric types.
  iree_host_size_t payload_candidate_count;
  // Number of payload-compatible descriptors that matched required flags.
  iree_host_size_t flag_candidate_count;
  // Number of flag-compatible descriptors available for the target features.
  iree_host_size_t feature_candidate_count;
} loom_cpu_packed_dot_match_diagnostic_t;

// Returns the stable display name for a packed-dot family.
iree_string_view_t loom_cpu_packed_dot_family_name(
    loom_cpu_packed_dot_family_t family);

// Returns the stable display name for a packed-dot numeric type.
iree_string_view_t loom_cpu_packed_dot_numeric_type_name(
    loom_cpu_packed_dot_numeric_type_t numeric_type);

// Maps a symbolic feature-set name such as "x86-avx-vnni" to feature bits.
iree_status_t loom_cpu_packed_dot_feature_bits_for_name(
    iree_string_view_t name,
    loom_cpu_packed_dot_feature_bits_t* out_feature_bits);

// Returns the number of built-in CPU packed-dot descriptors.
iree_host_size_t loom_cpu_packed_dot_descriptor_count(void);

// Returns a built-in descriptor by ordinal, or NULL when |index| is out of
// range.
const loom_cpu_packed_dot_descriptor_t* loom_cpu_packed_dot_descriptor_at(
    iree_host_size_t index);

// Finds a built-in descriptor by stable descriptor name.
const loom_cpu_packed_dot_descriptor_t* loom_cpu_packed_dot_find_by_name(
    iree_string_view_t name);

// Returns whether a descriptor is legal for a CPU feature set.
bool loom_cpu_packed_dot_is_available(
    const loom_cpu_packed_dot_descriptor_t* descriptor,
    loom_cpu_packed_dot_feature_bits_t feature_bits);

// Selects the first descriptor that satisfies a structural match request.
// Returns NULL when no descriptor matches and optionally populates
// |out_diagnostic| with the first filter that rejected all candidates.
const loom_cpu_packed_dot_descriptor_t* loom_cpu_packed_dot_select(
    const loom_cpu_packed_dot_match_request_t* request,
    loom_cpu_packed_dot_match_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_CPU_PACKED_DOT_CONTRACT_H_
