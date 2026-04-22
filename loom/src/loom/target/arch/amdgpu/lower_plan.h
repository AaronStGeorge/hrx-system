// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low lowering plans selected before emission.
//
// These structs are immutable emission contracts. The planner computes them
// once from source IR, facts, target profile, and descriptor availability; the
// emitter consumes them without re-running legality or descriptor selection.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_

#include <stdint.h>

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_bitfield_extract_plan_t {
  // Source vector value containing i32 lanes.
  loom_value_id_t source;
  // Result vector value containing i32 lanes.
  loom_value_id_t result;
  // Least-significant source bit of the extracted field.
  uint32_t offset;
  // Number of bits extracted from each lane.
  uint32_t width;
  // True when the extracted field is sign-extended.
  bool is_signed;
} loom_amdgpu_bitfield_extract_plan_t;

typedef struct loom_amdgpu_bitfield_insert_plan_t {
  // Field vector value containing i32 lanes.
  loom_value_id_t field;
  // Base vector value containing i32 lanes.
  loom_value_id_t base;
  // Result vector value containing i32 lanes.
  loom_value_id_t result;
  // Least-significant destination bit of the inserted field.
  uint32_t offset;
  // Number of low field bits inserted into each base lane.
  uint32_t width;
} loom_amdgpu_bitfield_insert_plan_t;

typedef struct loom_amdgpu_bitpack_plan_t {
  // Source vector value containing unpacked i32 lanes.
  loom_value_id_t source;
  // Result vector value containing packed i8 lanes.
  loom_value_id_t result;
  // Number of packed 32-bit registers in the result.
  uint32_t result_register_count;
} loom_amdgpu_bitpack_plan_t;

typedef struct loom_amdgpu_bitunpack_plan_t {
  // Source vector value containing packed integer bitstream storage.
  loom_value_id_t source;
  // Result vector value containing unpacked i32 lanes.
  loom_value_id_t result;
  // Number of bits unpacked into each result lane.
  uint32_t width;
  // Number of packed 32-bit source registers.
  uint32_t source_register_count;
  // Number of unpacked result lanes.
  uint32_t lane_count;
  // True when unpacked lanes are sign-extended.
  bool is_signed;
} loom_amdgpu_bitunpack_plan_t;

typedef struct loom_amdgpu_vector_bitcast_plan_t {
  // Source vector value being reinterpreted.
  loom_value_id_t source;
  // Result vector value receiving the same register payload.
  loom_value_id_t result;
} loom_amdgpu_vector_bitcast_plan_t;

typedef struct loom_amdgpu_vector_extract_plan_t {
  // Source vector value containing 32-bit lanes.
  loom_value_id_t source;
  // Result scalar value receiving one extracted lane.
  loom_value_id_t result;
  // Static source lane offset.
  uint32_t lane_offset;
  // Static source lane count.
  uint32_t lane_count;
} loom_amdgpu_vector_extract_plan_t;

typedef struct loom_amdgpu_buffer_alloca_plan_t {
  // Exact LDS allocation byte length proven during planning.
  int64_t byte_length;
  // Power-of-two LDS allocation byte alignment proven during planning.
  int64_t base_alignment;
} loom_amdgpu_buffer_alloca_plan_t;

typedef enum loom_amdgpu_table_index_kind_e {
  LOOM_AMDGPU_TABLE_INDEX_KIND_NONE = 0,
  LOOM_AMDGPU_TABLE_INDEX_KIND_I32 = 1,
  LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8 = 2,
} loom_amdgpu_table_index_kind_t;

typedef struct loom_amdgpu_table_lookup_plan_t {
  // Register table value selected by each index lane.
  loom_value_id_t table;
  // Index vector selecting table lanes.
  loom_value_id_t indices;
  // Result vector receiving selected table lanes.
  loom_value_id_t result;
  // Selected index payload representation.
  loom_amdgpu_table_index_kind_t index_kind;
  // Static number of table lanes.
  uint32_t table_lane_count;
  // Static number of result lanes.
  uint32_t result_lane_count;
  // Number of 32-bit registers occupied by the index vector.
  uint32_t index_register_count;
} loom_amdgpu_table_lookup_plan_t;

typedef struct loom_amdgpu_dot_plan_t {
  // Stable descriptor ID selected for the active descriptor set.
  uint64_t descriptor_id;
  // Number of scalar FMA lanes or packed dot register groups emitted.
  uint32_t iteration_count;
} loom_amdgpu_dot_plan_t;

typedef struct loom_amdgpu_vector_reduce_plan_t {
  // Stable descriptor ID selected for the reduction lane operation.
  uint64_t descriptor_id;
  // Source vector value being reduced.
  loom_value_id_t input;
  // Initial accumulator scalar value.
  loom_value_id_t init;
  // Result scalar value.
  loom_value_id_t result;
  // Scalar payload element type.
  loom_scalar_type_t element_type;
  // Static number of input lanes reduced.
  uint32_t lane_count;
} loom_amdgpu_vector_reduce_plan_t;

typedef struct loom_amdgpu_vector_compare_plan_t {
  // Stable descriptor ID selected for the compare predicate.
  uint64_t descriptor_id;
  // Left-hand payload vector value.
  loom_value_id_t lhs;
  // Right-hand payload vector value.
  loom_value_id_t rhs;
  // Result mask vector value.
  loom_value_id_t result;
  // Static number of payload and mask lanes compared.
  uint32_t lane_count;
} loom_amdgpu_vector_compare_plan_t;

typedef struct loom_amdgpu_vector_select_plan_t {
  // Source mask vector selecting true lanes.
  loom_value_id_t condition;
  // Source vector used when the corresponding condition lane is true.
  loom_value_id_t true_value;
  // Source vector used when the corresponding condition lane is false.
  loom_value_id_t false_value;
  // Result vector value.
  loom_value_id_t result;
  // Static number of selected 32-bit lanes.
  uint32_t lane_count;
} loom_amdgpu_vector_select_plan_t;

typedef enum loom_amdgpu_vector_slice_kind_e {
  LOOM_AMDGPU_VECTOR_SLICE_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES = 1,
  LOOM_AMDGPU_VECTOR_SLICE_KIND_PACKED_INTEGER = 2,
} loom_amdgpu_vector_slice_kind_t;

typedef struct loom_amdgpu_vector_slice_plan_t {
  // Source vector value being sliced.
  loom_value_id_t source;
  // Result vector value produced by the slice.
  loom_value_id_t result;
  // Selected lowering strategy for the source/result storage.
  loom_amdgpu_vector_slice_kind_t kind;
  // Static source lane offset.
  uint32_t lane_offset;
  // Static result lane count.
  uint32_t lane_count;
  // Source 32-bit backing register count.
  uint32_t source_register_count;
  // Result 32-bit backing register count.
  uint32_t result_register_count;
  // Source element bit count for packed integer slices.
  uint32_t element_bit_count;
} loom_amdgpu_vector_slice_plan_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE \
  LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE

typedef enum loom_amdgpu_memory_address_form_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR = 2,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR = 3,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID = 4,
} loom_amdgpu_memory_address_form_t;

typedef enum loom_amdgpu_memory_dynamic_index_kind_e {
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE = 0,
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR = 1,
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET = 2,
} loom_amdgpu_memory_dynamic_index_kind_t;

typedef enum loom_amdgpu_memory_operation_kind_e {
  LOOM_AMDGPU_MEMORY_OPERATION_LOAD = 0,
  LOOM_AMDGPU_MEMORY_OPERATION_STORE = 1,
} loom_amdgpu_memory_operation_kind_t;

typedef struct loom_amdgpu_memory_access_plan_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Selected target addressing form for the memory packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path used for the dynamic index.
  loom_amdgpu_memory_dynamic_index_kind_t dynamic_index_kind;
  // Static offset value encoded in the descriptor's first offset immediate.
  int64_t immediate_offset;
  // Static offset value encoded in the descriptor's second offset immediate.
  int64_t secondary_immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Number of 32-bit VGPR lanes moved by the selected memory packet.
  uint32_t vgpr_count;
  // Number of bytes moved by the selected memory packet.
  uint32_t packet_byte_count;
  // Stable descriptor ID selected for the active descriptor set.
  uint64_t descriptor_id;
} loom_amdgpu_memory_access_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_
