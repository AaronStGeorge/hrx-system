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

#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"

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

typedef struct loom_amdgpu_vector_compare_plan_t {
  // Descriptor row selected for the compare predicate.
  loom_low_lower_resolved_descriptor_t descriptor;
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

typedef enum loom_amdgpu_subgroup_payload_kind_e {
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE = 0,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR = 1,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR = 2,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR = 3,
  LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR = 4,
} loom_amdgpu_subgroup_payload_kind_t;

typedef struct loom_amdgpu_subgroup_broadcast_plan_t {
  // Descriptor row selected for the native cross-lane read.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Source value broadcast from source_lane.
  loom_value_id_t value;
  // Result value receiving the broadcast payload.
  loom_value_id_t result;
  // Source/result payload shape selected during planning.
  loom_amdgpu_subgroup_payload_kind_t payload_kind;
  // Number of 32-bit registers in the broadcast payload.
  uint32_t register_count;
  // Exact subgroup lane read by the broadcast.
  uint32_t source_lane;
} loom_amdgpu_subgroup_broadcast_plan_t;

typedef struct loom_amdgpu_matrix_mma_plan_t {
  // Descriptor row selected for the native matrix instruction.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Source matrix A fragment value.
  loom_value_id_t lhs;
  // Source matrix B fragment value.
  loom_value_id_t rhs;
  // Source matrix C accumulator fragment value.
  loom_value_id_t init;
  // Source matrix D result value.
  loom_value_id_t result;
} loom_amdgpu_matrix_mma_plan_t;

typedef enum loom_amdgpu_vector_slice_kind_e {
  LOOM_AMDGPU_VECTOR_SLICE_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES = 1,
  LOOM_AMDGPU_VECTOR_SLICE_KIND_PACKED_REGISTER_BITS = 2,
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
  // Source element bit count for packed register-bit slices.
  uint32_t element_bit_count;
} loom_amdgpu_vector_slice_plan_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE \
  LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE

typedef enum loom_amdgpu_memory_dynamic_index_kind_e {
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE = 0,
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR = 1,
  LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET = 2,
} loom_amdgpu_memory_dynamic_index_kind_t;

typedef enum loom_amdgpu_memory_operation_kind_e {
  LOOM_AMDGPU_MEMORY_OPERATION_LOAD = 0,
  LOOM_AMDGPU_MEMORY_OPERATION_STORE = 1,
} loom_amdgpu_memory_operation_kind_t;

typedef struct loom_amdgpu_memory_access_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Selected target addressing form for the memory packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static offset value encoded in the descriptor's first offset immediate.
  int64_t immediate_offset;
  // Static offset value encoded in the descriptor's second offset immediate.
  int64_t secondary_immediate_offset;
  // Static byte offset materialized through the VGPR VADDR operand.
  uint32_t vaddr_static_byte_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Number of 32-bit VGPR lanes moved by the selected memory packet.
  uint32_t vgpr_count;
  // Number of bytes moved by the selected memory packet.
  uint32_t packet_byte_count;
  // Descriptor row selected for the active descriptor set.
  const loom_low_descriptor_t* descriptor;
} loom_amdgpu_memory_access_t;

typedef struct loom_amdgpu_memory_access_plan_t {
  // Selected access form shared with legality and diagnostic consumers.
  loom_amdgpu_memory_access_t access;
  // Module string ID for access.descriptor's opcode spelling.
  loom_string_id_t opcode_id;
} loom_amdgpu_memory_access_plan_t;

#define LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY 4

typedef struct loom_amdgpu_explicit_packet_immediate_t {
  // Module string ID for the immediate field name.
  loom_string_id_t name_id;
  // Concrete immediate value emitted for the packet.
  uint16_t value;
} loom_amdgpu_explicit_packet_immediate_t;

typedef struct loom_amdgpu_explicit_packet_immediate_template_t {
  // Borrowed immediate field name resolved during packet planning.
  iree_string_view_t name;
  // Concrete immediate value emitted for the packet.
  uint16_t value;
} loom_amdgpu_explicit_packet_immediate_template_t;

typedef struct loom_amdgpu_explicit_packet_plan_t {
  // Descriptor row selected for the explicit packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Immediate rows emitted on the descriptor.
  loom_amdgpu_explicit_packet_immediate_t
      immediates[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_explicit_packet_plan_t;

#define LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY 2
#define LOOM_AMDGPU_ATOMIC_CACHE_CONTROL_CAPACITY 2

typedef uint32_t loom_amdgpu_atomic_packet_attr_flags_t;

#define LOOM_AMDGPU_ATOMIC_PACKET_ATTR_SCOPE ((uint32_t)1u << 0)

typedef struct loom_amdgpu_atomic_packet_attrs_t {
  // Attribute bits populated for the selected atomic packet.
  loom_amdgpu_atomic_packet_attr_flags_t flags;
  // Module string ID for the scope attribute when present.
  loom_string_id_t scope_attr_name_id;
  // VGLOBAL SCOPE immediate value encoded on GFX12 atomic packets.
  int64_t scope;
} loom_amdgpu_atomic_packet_attrs_t;

typedef struct loom_amdgpu_atomic_ordering_plan_t {
  // Explicit waits emitted before the atomic packet.
  loom_amdgpu_explicit_packet_plan_t
      pre_atomic_waits[LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY];
  // Number of populated pre-atomic wait packets.
  iree_host_size_t pre_atomic_wait_count;
  // Explicit waits emitted after the atomic packet.
  loom_amdgpu_explicit_packet_plan_t
      post_atomic_waits[LOOM_AMDGPU_ATOMIC_WAIT_CAPACITY];
  // Number of populated post-atomic wait packets.
  iree_host_size_t post_atomic_wait_count;
  // Explicit cache controls emitted after the atomic packet.
  loom_amdgpu_explicit_packet_plan_t
      post_atomic_cache_controls[LOOM_AMDGPU_ATOMIC_CACHE_CONTROL_CAPACITY];
  // Number of populated post-atomic cache-control packets.
  iree_host_size_t post_atomic_cache_control_descriptor_count;
} loom_amdgpu_atomic_ordering_plan_t;

typedef uint32_t loom_amdgpu_atomic_plan_flags_t;

#define LOOM_AMDGPU_ATOMIC_PLAN_REQUIRES_M0 ((uint32_t)1u << 0)

typedef struct loom_amdgpu_atomic_plan_t {
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Target-specific lowering flags derived from the selected descriptor.
  loom_amdgpu_atomic_plan_flags_t flags;
  // Source atomic operation form being lowered.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Selected target addressing form for the atomic packet.
  loom_amdgpu_memory_address_form_t address_form;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static offset value encoded in the descriptor offset immediate.
  int64_t immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Descriptor row selected for the active descriptor set.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Descriptor attrs emitted directly on the selected atomic packet.
  loom_amdgpu_atomic_packet_attrs_t packet_attrs;
  // Explicit packets required to implement source atomic ordering.
  loom_amdgpu_atomic_ordering_plan_t ordering;
} loom_amdgpu_atomic_plan_t;

typedef struct loom_amdgpu_prefetch_plan_t {
  // Descriptor row selected for the prefetch packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's offset attribute.
  loom_string_id_t offset_attr_name_id;
  // Module string ID for the descriptor's count attribute.
  loom_string_id_t count_attr_name_id;
  // Target-independent source memory access plan being wrapped.
  loom_low_source_memory_access_plan_t source;
  // Target operand path selected for the source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t dynamic_term_kind;
  // Static offset value encoded in the descriptor offset immediate.
  int64_t immediate_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Prefetch span count encoded in the descriptor count immediate.
  uint32_t count;
} loom_amdgpu_prefetch_plan_t;

typedef struct loom_amdgpu_async_gather_plan_t {
  // Source global-like view access transferred into LDS.
  loom_low_source_memory_access_plan_t source;
  // Target operand path selected for each source dynamic address term.
  loom_amdgpu_memory_dynamic_index_kind_t
      source_dynamic_term_kinds[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Source SSA view value passed to kernel.async.gather.
  loom_value_id_t source_view;
  // Destination LDS view value passed to kernel.async.gather.
  loom_value_id_t dest_view;
  // Static LDS byte offset materialized into M0.
  uint32_t dest_byte_offset;
  // Static global byte offset encoded in the packet immediate.
  int64_t source_immediate_offset;
  // Number of bytes moved by the selected async packet.
  uint32_t packet_byte_count;
  // Descriptor row selected for the active descriptor set.
  loom_low_lower_resolved_descriptor_t descriptor;
} loom_amdgpu_async_gather_plan_t;

#define LOOM_AMDGPU_ASYNC_WAIT_IMMEDIATE_CAPACITY \
  LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY

typedef loom_amdgpu_explicit_packet_immediate_template_t
    loom_amdgpu_async_wait_immediate_t;

typedef struct loom_amdgpu_async_wait_plan_t {
  // Explicit wait packet selected for the async stream wait.
  loom_amdgpu_explicit_packet_plan_t wait;
} loom_amdgpu_async_wait_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_PLAN_H_
