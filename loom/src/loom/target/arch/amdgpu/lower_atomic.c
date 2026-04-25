// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/context.h"
#include "loom/ops/atomic.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/target/arch/amdgpu/lower_memory_internal.h"
#include "loom/target/arch/amdgpu/target_info.h"

typedef uint32_t loom_amdgpu_atomic_rejection_flags_t;

#define LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP ((uint32_t)1u << 0)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND ((uint32_t)1u << 1)
#define LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE ((uint32_t)1u << 2)
#define LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT ((uint32_t)1u << 3)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND ((uint32_t)1u << 5)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE ((uint32_t)1u << 6)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT ((uint32_t)1u << 7)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING ((uint32_t)1u << 8)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY ((uint32_t)1u << 10)
#define LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING ((uint32_t)1u << 11)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE ((uint32_t)1u << 12)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE ((uint32_t)1u << 13)

typedef struct loom_amdgpu_atomic_diagnostic_t {
  // Rejection bits explaining why a source atomic is not legal.
  loom_amdgpu_atomic_rejection_flags_t rejection_bits;
} loom_amdgpu_atomic_diagnostic_t;

typedef struct loom_amdgpu_atomic_rejection_detail_t {
  // Rejection bit matched by this diagnostic row.
  loom_amdgpu_atomic_rejection_flags_t rejection_bit;
  // User-facing diagnostic detail for the matched rejection bit.
  iree_string_view_t detail;
} loom_amdgpu_atomic_rejection_detail_t;

typedef struct loom_amdgpu_atomic_wait_immediate_template_t {
  // Immediate field name emitted on the wait packet.
  iree_string_view_t name;
  // Immediate value emitted on the wait packet.
  uint16_t value;
} loom_amdgpu_atomic_wait_immediate_template_t;

typedef struct loom_amdgpu_atomic_wait_packet_template_t {
  // Stable descriptor ID emitted for this wait packet.
  uint64_t descriptor_id;
  // Static immediate rows for draining the selected counter to zero.
  loom_amdgpu_atomic_wait_immediate_template_t
      immediates[LOOM_AMDGPU_EXPLICIT_WAIT_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_atomic_wait_packet_template_t;

typedef enum loom_amdgpu_atomic_value_kind_e {
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32 = 0,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32 = 1,
} loom_amdgpu_atomic_value_kind_t;

#define LOOM_AMDGPU_ATOMIC_KIND_NONE UINT8_MAX

typedef struct loom_amdgpu_atomic_descriptor_candidate_t {
  // Source memory space matched by this row.
  loom_value_fact_memory_space_t memory_space;
  // Target addressing form emitted by this row.
  loom_amdgpu_memory_address_form_t address_form;
  // Source atomic operation form matched by this row.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Source atomic arithmetic kind matched by this row.
  uint8_t atomic_kind;
  // Source scalar value type required by this row.
  loom_amdgpu_atomic_value_kind_t value_kind;
  // Stable descriptor ID selected when present in the descriptor set.
  uint64_t descriptor_id;
} loom_amdgpu_atomic_descriptor_candidate_t;

#define LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE(memory_space_, address_form_,  \
                                                operation_kind_, atomic_kind_, \
                                                value_kind_, descriptor_id_)   \
  {                                                                            \
      .memory_space = memory_space_,                                           \
      .address_form = address_form_,                                           \
      .operation_kind = operation_kind_,                                       \
      .atomic_kind = atomic_kind_,                                             \
      .value_kind = value_kind_,                                               \
      .descriptor_id = descriptor_id_,                                         \
  }

#define LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(                          \
    operation_kind_, atomic_kind_, value_kind_, descriptor_id_)               \
  LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE(                                    \
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,                                 \
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, operation_kind_, atomic_kind_, \
      value_kind_, descriptor_id_)

#define LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(        \
    operation_kind_, atomic_kind_, value_kind_, descriptor_id_)      \
  LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE(                           \
      LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,                           \
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR, operation_kind_, \
      atomic_kind_, value_kind_, descriptor_id_)

#define LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(                       \
    operation_kind_, atomic_kind_, value_kind_, descriptor_id_)               \
  LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE(                                    \
      LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,                                    \
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, operation_kind_, atomic_kind_, \
      value_kind_, descriptor_id_)

static const loom_amdgpu_atomic_descriptor_candidate_t
    kAmdgpuAtomicDescriptorCandidates[] = {
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ADDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_SUBI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_SUB_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MINSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MIN_I32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MAXSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MAX_I32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MINUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MIN_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MAXUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MAX_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ANDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_AND_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_OR_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_XORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_XOR_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ADDF,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_F32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ADDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_RTN_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_SUBI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_SUB_RTN_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MINSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MIN_RTN_I32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MAXSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MAX_RTN_I32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MINUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MIN_RTN_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MAXUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_MAX_RTN_U32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ANDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_AND_RTN_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_OR_RTN_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_XORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_XOR_RTN_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_XCHGI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRXCHG_RTN_B32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ADDF,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_RTN_F32),
        LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG, LOOM_AMDGPU_ATOMIC_KIND_NONE,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_DS_CMPST_RTN_B32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ADDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_ADD_U32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_SUBI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_SUB_U32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MINSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MIN_I32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MAXSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MAX_I32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MINUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MIN_U32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MAXUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MAX_U32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ANDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_AND_B32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_OR_B32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_XORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_XOR_B32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ADDF,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_ADD_F32),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ADDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_ADD_U32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_SUBI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_SUB_U32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MINSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MIN_I32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MAXSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MAX_I32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MINUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MIN_U32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MAXUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_MAX_U32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ANDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_AND_B32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_OR_B32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_XORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_XOR_B32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_XCHGI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_SWAP_B32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ADDF,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_ADD_F32_RTN),
        LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG, LOOM_AMDGPU_ATOMIC_KIND_NONE,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_ATOMIC_CMPSWAP_B32_RTN),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ADDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_ADD_U32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_SUBI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_SUB_U32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MINSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MIN_I32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MAXSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MAX_I32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MINUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MIN_U32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_MAXUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MAX_U32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ANDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_AND_B32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_OR_B32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_XORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_XOR_B32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE, LOOM_ATOMIC_KIND_ADDF,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_ADD_F32_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ADDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_ADD_U32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_SUBI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_SUB_U32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MINSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MIN_I32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MAXSI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MAX_I32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MINUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MIN_U32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_MAXUI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_MAX_U32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ANDI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_AND_B32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_OR_B32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_XORI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_XOR_B32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_XCHGI,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_SWAP_B32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_RMW, LOOM_ATOMIC_KIND_ADDF,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_ADD_F32_RTN_SADDR),
        LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE(
            LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG, LOOM_AMDGPU_ATOMIC_KIND_NONE,
            LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_ATOMIC_CMPSWAP_B32_RTN_SADDR),
};

#undef LOOM_AMDGPU_BUFFER_ATOMIC_DESCRIPTOR_CANDIDATE
#undef LOOM_AMDGPU_GLOBAL_SADDR_ATOMIC_DESCRIPTOR_CANDIDATE
#undef LOOM_AMDGPU_LDS_ATOMIC_DESCRIPTOR_CANDIDATE
#undef LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE

static const loom_amdgpu_atomic_rejection_detail_t
    kAmdgpuAtomicRejectionDetails[] = {
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP,
            .detail = IREE_SVL("source op is not a supported atomic access"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND,
            .detail =
                IREE_SVL("source atomic operation kind is not representable"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE,
            .detail = IREE_SVL(
                "AMDGPU atomic lowering currently supports workgroup and "
                "global memory"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic lowering requires an LDS buffer root"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE,
            .detail =
                IREE_SVL("AMDGPU atomic lowering requires one 32-bit element"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND,
            .detail = IREE_SVL("AMDGPU atomic lowering currently supports "
                               "32-bit add/sub/min/max/and/or/xor/exchange/"
                               "compare-exchange and f32 add"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE,
            .detail = IREE_SVL(
                "AMDGPU atomic kind does not match the source value type"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT,
            .detail =
                IREE_SVL("AMDGPU atomic value must be available as a VGPR"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING,
            .detail = IREE_SVL("AMDGPU atomic lowering currently supports "
                               "relaxed global atomics, GFX11 global "
                               "acquire/release atomics, and "
                               "workgroup-scope LDS atomics"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE,
            .detail = IREE_SVL("AMDGPU atomic lowering currently supports "
                               "workgroup scope for workgroup memory and "
                               "device scope for global memory"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY,
            .detail = IREE_SVL(
                "AMDGPU atomic lowering does not support cache policies"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING,
            .detail = IREE_SVL("selected descriptor set does not provide an "
                               "atomic packet"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE,
            .detail = IREE_SVL(
                "AMDGPU atomic descriptor has no usable offset immediate"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE,
            .detail = IREE_SVL("AMDGPU atomic static offset cannot be encoded"),
        },
};

// GFX11 VMEM load drain used by global atomic acquire/release ordering.
static const loom_amdgpu_atomic_wait_packet_template_t
    kAmdgpuGfx11VmemLoadWaitPacket = {
        .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_WAITCNT,
        .immediates =
            {
                {IREE_SVL("vmcnt"), 0},
                {IREE_SVL("lgkmcnt"), 15},
            },
        .immediate_count = 2,
};

// GFX11 VMEM store drain used by global atomic acquire/release ordering.
static const loom_amdgpu_atomic_wait_packet_template_t
    kAmdgpuGfx11VmemStoreWaitPacket = {
        .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_WAITCNT_VSCNT,
        .immediates =
            {
                {IREE_SVL("vscnt"), 0},
            },
        .immediate_count = 1,
};

static bool loom_amdgpu_view_atomic_isa(const loom_op_t* op) {
  return loom_view_atomic_reduce_isa(op) || loom_view_atomic_rmw_isa(op) ||
         loom_view_atomic_cmpxchg_isa(op);
}

static loom_value_id_t loom_amdgpu_atomic_value(const loom_op_t* op) {
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_value(op);
  }
  return loom_view_atomic_rmw_value(op);
}

static uint8_t loom_amdgpu_atomic_kind(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return LOOM_AMDGPU_ATOMIC_KIND_NONE;
  }
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_kind(op);
  }
  return loom_view_atomic_rmw_kind(op);
}

static uint8_t loom_amdgpu_atomic_ordering(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return loom_view_atomic_cmpxchg_success_ordering(op);
  }
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_ordering(op);
  }
  return loom_view_atomic_rmw_ordering(op);
}

static uint8_t loom_amdgpu_atomic_failure_ordering(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return loom_view_atomic_cmpxchg_failure_ordering(op);
  }
  return loom_amdgpu_atomic_ordering(op);
}

static uint8_t loom_amdgpu_atomic_scope(const loom_op_t* op) {
  if (loom_view_atomic_cmpxchg_isa(op)) {
    return loom_view_atomic_cmpxchg_scope(op);
  }
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_scope(op);
  }
  return loom_view_atomic_rmw_scope(op);
}

static bool loom_amdgpu_atomic_descriptor_available(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t descriptor_id) {
  return loom_low_descriptor_set_lookup_descriptor_by_id(
             descriptor_set, descriptor_id) != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

static bool loom_amdgpu_atomic_ordering_has_acquire(uint8_t ordering) {
  switch (ordering) {
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_ordering_has_release(uint8_t ordering) {
  switch (ordering) {
    case LOOM_ATOMIC_ORDERING_RELEASE:
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_atomic_has_acquire_ordering(const loom_op_t* op) {
  return loom_amdgpu_atomic_ordering_has_acquire(
             loom_amdgpu_atomic_ordering(op)) ||
         loom_amdgpu_atomic_ordering_has_acquire(
             loom_amdgpu_atomic_failure_ordering(op));
}

static bool loom_amdgpu_atomic_has_release_ordering(const loom_op_t* op) {
  return loom_amdgpu_atomic_ordering_has_release(
      loom_amdgpu_atomic_ordering(op));
}

static bool loom_amdgpu_atomic_global_ordering_supported(
    const loom_low_descriptor_set_t* descriptor_set, uint8_t ordering) {
  if (ordering != LOOM_ATOMIC_ORDERING_ACQUIRE &&
      ordering != LOOM_ATOMIC_ORDERING_RELEASE &&
      ordering != LOOM_ATOMIC_ORDERING_ACQ_REL &&
      ordering != LOOM_ATOMIC_ORDERING_SEQ_CST) {
    return false;
  }
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_by_id(descriptor_set->stable_id);
  return descriptor_set_info != NULL &&
         descriptor_set_info->vector_memory_cache_policy_encoding ==
             LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC;
}

static bool loom_amdgpu_atomic_ordering_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space, uint8_t ordering) {
  if (ordering == LOOM_ATOMIC_ORDERING_RELAXED) {
    return true;
  }
  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    return loom_amdgpu_atomic_global_ordering_supported(descriptor_set,
                                                        ordering);
  }
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  return ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
         ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
         ordering == LOOM_ATOMIC_ORDERING_ACQ_REL ||
         ordering == LOOM_ATOMIC_ORDERING_SEQ_CST;
}

static bool loom_amdgpu_atomic_orderings_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_memory_space_t memory_space, const loom_op_t* op) {
  return loom_amdgpu_atomic_ordering_supported(
             descriptor_set, memory_space, loom_amdgpu_atomic_ordering(op)) &&
         loom_amdgpu_atomic_ordering_supported(
             descriptor_set, memory_space,
             loom_amdgpu_atomic_failure_ordering(op));
}

static bool loom_amdgpu_atomic_value_kind_matches(
    loom_type_t value_type, loom_amdgpu_atomic_value_kind_t value_kind) {
  switch (value_kind) {
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32:
      return loom_amdgpu_type_is_i32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32:
      return loom_amdgpu_type_is_f32(value_type);
  }
  return false;
}

static bool loom_amdgpu_atomic_value_can_feed_vgpr_operand(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_amdgpu_atomic_descriptor_candidate_t* candidate) {
  if (candidate->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    int64_t unused_value = 0;
    const loom_value_id_t expected =
        loom_view_atomic_cmpxchg_expected(source_op);
    const loom_value_id_t replacement =
        loom_view_atomic_cmpxchg_replacement(source_op);
    return (loom_amdgpu_module_value_prefers_vgpr(module, expected) ||
            loom_amdgpu_module_value_as_i32_constant(module, expected,
                                                     &unused_value)) &&
           (loom_amdgpu_module_value_prefers_vgpr(module, replacement) ||
            loom_amdgpu_module_value_as_i32_constant(module, replacement,
                                                     &unused_value));
  }
  if (candidate->value_kind == LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32) {
    return true;
  }
  const loom_value_id_t value_id = loom_amdgpu_atomic_value(source_op);
  int64_t unused_value = 0;
  return loom_amdgpu_module_value_prefers_vgpr(module, value_id) ||
         loom_amdgpu_module_value_as_i32_constant(module, value_id,
                                                  &unused_value);
}

static bool loom_amdgpu_atomic_source_plan_proves_workgroup_root(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source) {
  if (source->root_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* root_value =
      loom_module_value(module, source->root_value_id);
  if (loom_value_is_block_arg(root_value)) {
    return false;
  }
  const loom_op_t* root_op = loom_value_def_op(root_value);
  return root_op != NULL && loom_buffer_alloca_isa(root_op);
}

static bool loom_amdgpu_atomic_select_descriptor(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* plan, loom_type_t value_type,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  plan->descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  const uint8_t atomic_kind = loom_amdgpu_atomic_kind(source_op);
  bool found_kind = false;
  bool found_type = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuAtomicDescriptorCandidates); ++i) {
    const loom_amdgpu_atomic_descriptor_candidate_t* candidate =
        &kAmdgpuAtomicDescriptorCandidates[i];
    if (candidate->memory_space != plan->source.memory_space ||
        candidate->operation_kind != plan->operation_kind) {
      continue;
    }
    if (plan->operation_kind != LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG &&
        candidate->atomic_kind != atomic_kind) {
      continue;
    }
    found_kind = true;
    if (!loom_amdgpu_atomic_value_kind_matches(value_type,
                                               candidate->value_kind)) {
      continue;
    }
    found_type = true;
    if (!loom_amdgpu_atomic_value_can_feed_vgpr_operand(module, source_op,
                                                        candidate)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT;
      return false;
    }
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(
            descriptor_set, candidate->descriptor_id);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    plan->address_form = candidate->address_form;
    plan->descriptor_id = candidate->descriptor_id;
    return true;
  }
  if (found_type) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
  } else {
    diagnostic->rejection_bits |=
        found_kind ? LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE
                   : LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND;
  }
  return false;
}

static bool loom_amdgpu_atomic_uses_buffer_resource(
    const loom_amdgpu_atomic_plan_t* plan) {
  return plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
         plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
}

static bool loom_amdgpu_atomic_select_offset(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t descriptor_id,
    loom_amdgpu_atomic_plan_t* plan,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  const loom_low_immediate_kind_t expected_kind =
      plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR
          ? LOOM_LOW_IMMEDIATE_KIND_SIGNED
          : LOOM_LOW_IMMEDIATE_KIND_UNSIGNED;
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, expected_kind, &offset_info) ||
      offset_info.unit_byte_count == 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE;
    return false;
  }
  if (expected_kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    const int64_t signed_max = offset_info.unsigned_max > INT64_MAX
                                   ? INT64_MAX
                                   : (int64_t)offset_info.unsigned_max;
    if (offset_info.unit_byte_count != 1 ||
        plan->source.static_byte_offset < offset_info.signed_min ||
        plan->source.static_byte_offset > signed_max) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    plan->immediate_offset = plan->source.static_byte_offset;
    plan->scalar_byte_offset = 0;
    return true;
  }
  if (plan->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t static_byte_offset = (uint64_t)plan->source.static_byte_offset;
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    if (offset_info.unit_byte_count != 1) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    const uint64_t immediate_offset =
        iree_min(static_byte_offset, offset_info.unsigned_max);
    const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
    if (scalar_byte_offset > UINT32_MAX) {
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
      return false;
    }
    plan->immediate_offset = (int64_t)immediate_offset;
    plan->scalar_byte_offset = (uint32_t)scalar_byte_offset;
    return true;
  }
  if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t encoded_offset =
      static_byte_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  plan->immediate_offset = (int64_t)encoded_offset;
  plan->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_atomic_append_wait_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_atomic_wait_packet_template_t* wait_packet,
    loom_amdgpu_explicit_wait_plan_t* waits, iree_host_size_t wait_capacity,
    iree_host_size_t* inout_wait_count) {
  if (!loom_amdgpu_atomic_descriptor_available(descriptor_set,
                                               wait_packet->descriptor_id)) {
    return false;
  }
  IREE_ASSERT(*inout_wait_count < wait_capacity);
  loom_amdgpu_explicit_wait_plan_t* wait = &waits[(*inout_wait_count)++];
  *wait = (loom_amdgpu_explicit_wait_plan_t){
      .descriptor_id = wait_packet->descriptor_id,
      .immediate_count = wait_packet->immediate_count,
  };
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    wait->immediates[i] = (loom_amdgpu_explicit_wait_immediate_t){
        .name = wait_packet->immediates[i].name,
        .value = wait_packet->immediates[i].value,
    };
  }
  return true;
}

static bool loom_amdgpu_atomic_select_global_release_waits(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_atomic_ordering_plan_t* ordering) {
  if (!loom_amdgpu_atomic_append_wait_packet(
          descriptor_set, &kAmdgpuGfx11VmemLoadWaitPacket,
          ordering->pre_atomic_waits,
          IREE_ARRAYSIZE(ordering->pre_atomic_waits),
          &ordering->pre_atomic_wait_count)) {
    return false;
  }
  return loom_amdgpu_atomic_append_wait_packet(
      descriptor_set, &kAmdgpuGfx11VmemStoreWaitPacket,
      ordering->pre_atomic_waits, IREE_ARRAYSIZE(ordering->pre_atomic_waits),
      &ordering->pre_atomic_wait_count);
}

static bool loom_amdgpu_atomic_select_global_acquire_waits(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_atomic_ordering_plan_t* ordering,
    loom_amdgpu_atomic_operation_kind_t operation_kind) {
  const loom_amdgpu_atomic_wait_packet_template_t* wait_packet =
      operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
          ? &kAmdgpuGfx11VmemStoreWaitPacket
          : &kAmdgpuGfx11VmemLoadWaitPacket;
  return loom_amdgpu_atomic_append_wait_packet(
      descriptor_set, wait_packet, ordering->post_atomic_waits,
      IREE_ARRAYSIZE(ordering->post_atomic_waits),
      &ordering->post_atomic_wait_count);
}

static bool loom_amdgpu_atomic_select_global_ordering(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_atomic_plan_t* plan, const loom_op_t* source_op,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  plan->ordering = (loom_amdgpu_atomic_ordering_plan_t){0};
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
      (!loom_amdgpu_atomic_has_release_ordering(source_op) &&
       !loom_amdgpu_atomic_has_acquire_ordering(source_op))) {
    return true;
  }

  if (loom_amdgpu_atomic_has_release_ordering(source_op)) {
    if (!loom_amdgpu_atomic_select_global_release_waits(descriptor_set,
                                                        &plan->ordering)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
  }
  if (loom_amdgpu_atomic_has_acquire_ordering(source_op)) {
    if (!loom_amdgpu_atomic_select_global_acquire_waits(
            descriptor_set, &plan->ordering, plan->operation_kind)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
    if (!loom_amdgpu_atomic_descriptor_available(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_GL1_INV) ||
        !loom_amdgpu_atomic_descriptor_available(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_GL0_INV)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
    plan->ordering.post_atomic_cache_control_descriptor_ids[0] =
        LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_GL1_INV;
    plan->ordering.post_atomic_cache_control_descriptor_ids[1] =
        LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_GL0_INV;
    plan->ordering.post_atomic_cache_control_descriptor_count = 2;
  }
  return true;
}

static bool loom_amdgpu_atomic_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* memory_diagnostic,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(source_diagnostic);
  IREE_ASSERT_ARGUMENT(memory_diagnostic);
  IREE_ASSERT_ARGUMENT(diagnostic);
  *out_plan = (loom_amdgpu_atomic_plan_t){
      .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
  };
  *source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *memory_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  *diagnostic = (loom_amdgpu_atomic_diagnostic_t){0};
  if (!loom_amdgpu_view_atomic_isa(source_op)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP;
    return false;
  }

  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &out_plan->source,
                                                source_diagnostic)) {
    return false;
  }
  switch (out_plan->source.operation_kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE:
      out_plan->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE;
      break;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW:
      out_plan->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_RMW;
      break;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG:
      out_plan->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG;
      break;
    default:
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND;
      return false;
  }
  switch (out_plan->source.memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      out_plan->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      if (!loom_amdgpu_atomic_source_plan_proves_workgroup_root(
              module, &out_plan->source)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT;
        return false;
      }
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      out_plan->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
      break;
    default:
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE;
      return false;
  }
  if (out_plan->source.element_byte_count != 4 ||
      out_plan->source.vector_lane_count != 1 ||
      out_plan->source.vector_lane_byte_stride != 4) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE;
    return false;
  }
  if (loom_amdgpu_memory_cache_policy_is_present(
          &out_plan->source.cache_policy)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY;
    return false;
  }
  if (!loom_amdgpu_atomic_orderings_supported(
          descriptor_set, out_plan->source.memory_space, source_op)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING;
    return false;
  }
  const uint8_t expected_scope =
      out_plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? LOOM_VIEW_SCOPE_WORKGROUP
          : LOOM_VIEW_SCOPE_DEVICE;
  if (loom_amdgpu_atomic_scope(source_op) != expected_scope) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE;
    return false;
  }

  loom_amdgpu_memory_access_plan_t memory_access = {
      .source = out_plan->source,
      .address_form = out_plan->address_form,
  };
  if (!loom_amdgpu_memory_access_select_dynamic_index_kind(
          module, &memory_access, memory_diagnostic)) {
    return false;
  }
  out_plan->dynamic_index_kind = memory_access.dynamic_index_kind;

  const loom_type_t value_type =
      out_plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG
          ? loom_module_value_type(module,
                                   loom_view_atomic_cmpxchg_old(source_op))
          : loom_module_value_type(module, loom_amdgpu_atomic_value(source_op));
  if (!loom_amdgpu_atomic_select_descriptor(module, descriptor_set, source_op,
                                            out_plan, value_type, diagnostic)) {
    return false;
  }
  if (out_plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR &&
      out_plan->dynamic_index_kind ==
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
    memory_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS;
    return false;
  }
  if (!loom_amdgpu_atomic_select_global_ordering(descriptor_set, out_plan,
                                                 source_op, diagnostic)) {
    return false;
  }
  return loom_amdgpu_atomic_select_offset(
      descriptor_set, out_plan->descriptor_id, out_plan, diagnostic);
}

static iree_string_view_t loom_amdgpu_atomic_rejection_detail(
    loom_amdgpu_atomic_rejection_flags_t rejection_bits) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuAtomicRejectionDetails); ++i) {
    const loom_amdgpu_atomic_rejection_detail_t* row =
        &kAmdgpuAtomicRejectionDetails[i];
    if (iree_any_bit_set(rejection_bits, row->rejection_bit)) {
      return row->detail;
    }
  }
  return IREE_SV("source atomic is not representable");
}

bool loom_amdgpu_select_view_atomic_plan(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_amdgpu_atomic_plan_t* out_plan) {
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  return loom_amdgpu_atomic_select(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan,
      &source_diagnostic, &memory_diagnostic, &diagnostic);
}

static iree_status_t loom_amdgpu_lookup_atomic_value_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t source_value = loom_amdgpu_atomic_value(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source_value);
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, source_value, out_low_value);
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU atomic selected a non-VGPR dynamic update value");
}

static iree_status_t loom_amdgpu_lookup_atomic_cmpxchg_values_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_expected, loom_value_id_t* out_low_replacement) {
  IREE_ASSERT_ARGUMENT(out_low_expected);
  IREE_ASSERT_ARGUMENT(out_low_replacement);
  *out_low_expected = LOOM_VALUE_ID_INVALID;
  *out_low_replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, loom_view_atomic_cmpxchg_expected(source_op),
      out_low_expected));
  return loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, loom_view_atomic_cmpxchg_replacement(source_op),
      out_low_replacement);
}

static iree_status_t loom_amdgpu_emit_atomic_cmpxchg_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_expected, loom_value_id_t low_replacement,
    loom_type_t pair_type, loom_value_id_t* out_low_pair) {
  IREE_ASSERT_ARGUMENT(out_low_pair);
  *out_low_pair = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {low_expected, low_replacement};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), operands,
      IREE_ARRAYSIZE(operands), pair_type, source_op->location, &concat_op));
  *out_low_pair = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_buffer_soffset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan, loom_value_id_t* out_low_soffset) {
  IREE_ASSERT_ARGUMENT(out_low_soffset);
  const loom_value_id_t dynamic_index =
      plan->dynamic_index_kind == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET
          ? plan->source.dynamic_index
          : LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_sgpr_byte_offset(
      context, source_op, dynamic_index, plan->source.dynamic_index_byte_stride,
      plan->source.dynamic_index_byte_shift, plan->scalar_byte_offset,
      out_low_soffset);
}

static iree_status_t loom_amdgpu_emit_atomic_ordering_waits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_wait_plan_t* waits,
    iree_host_size_t wait_count) {
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_explicit_wait_plan(context, source_op, &waits[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_cache_controls(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const uint64_t* descriptor_ids, iree_host_size_t descriptor_count) {
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, descriptor_ids[i], /*operands=*/NULL,
        /*operand_count=*/0, loom_make_named_attr_slice(NULL, 0),
        /*result_types=*/NULL, /*result_count=*/0, &low_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_atomic_post_ordering(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_ordering_plan_t* ordering) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_ordering_waits(
      context, source_op, ordering->post_atomic_waits,
      ordering->post_atomic_wait_count));
  return loom_amdgpu_emit_atomic_cache_controls(
      context, source_op, ordering->post_atomic_cache_control_descriptor_ids,
      ordering->post_atomic_cache_control_descriptor_count);
}

iree_status_t loom_amdgpu_lower_view_atomic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  if (plan->operation_kind != LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_value_as_vgpr(
        context, source_op, &low_value));
  }

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->source.view_value_id, &low_resource));

  loom_amdgpu_memory_access_plan_t access = {
      .source = plan->source,
      .address_form = plan->address_form,
      .dynamic_index_kind = plan->dynamic_index_kind,
      .immediate_offset = plan->immediate_offset,
      .scalar_byte_offset = plan->scalar_byte_offset,
      .vgpr_count = 1,
      .packet_byte_count = 4,
      .descriptor_id = plan->descriptor_id,
  };
  const loom_value_id_t low_base_addr =
      plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? low_resource
          : LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, &access, low_base_addr, &low_vaddr));

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, &access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_named_attr_slice_t packet_attrs =
      loom_make_named_attr_slice(attrs, attr_count);

  loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
  if (plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, low_resource, &low_saddr));
  }
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_buffer_soffset(
        context, source_op, plan, &low_soffset));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_ordering_waits(
      context, source_op, plan->ordering.pre_atomic_waits,
      plan->ordering.pre_atomic_wait_count));

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG) {
    loom_value_id_t low_expected = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_replacement = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_atomic_cmpxchg_values_as_vgpr(
        context, source_op, &low_expected, &low_replacement));
    loom_type_t old_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &old_type));
    loom_value_id_t low_old = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      loom_type_t pair_type = loom_type_none();
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_vgpr_range_type(context, 2, &pair_type));
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_cmpxchg_pair(
          context, source_op, low_expected, low_replacement, pair_type,
          &low_pair));
      loom_value_id_t operands[] = {low_pair, low_resource, low_vaddr,
                                    low_soffset};
      const loom_tied_result_t tied_result = {
          .result_index = 0,
          .operand_index = 0,
      };
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
          context, plan->descriptor_id, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &pair_type, 1, &tied_result, 1, source_op->location,
          &low_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op,
          loom_value_slice_get(loom_low_op_results(low_op), 0), 0, old_type,
          &low_old));
    } else if (plan->address_form ==
               LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
      loom_type_t pair_type = loom_type_none();
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_vgpr_range_type(context, 2, &pair_type));
      loom_value_id_t low_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_cmpxchg_pair(
          context, source_op, low_expected, low_replacement, pair_type,
          &low_pair));
      loom_value_id_t operands[] = {low_vaddr, low_pair, low_saddr};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, plan->descriptor_id, operands,
          IREE_ARRAYSIZE(operands), packet_attrs, &old_type, 1, &low_op));
      low_old = loom_value_slice_get(loom_low_op_results(low_op), 0);
    } else {
      loom_value_id_t operands[] = {low_vaddr, low_expected, low_replacement};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, plan->descriptor_id, operands,
          IREE_ARRAYSIZE(operands), packet_attrs, &old_type, 1, &low_op));
      low_old = loom_value_slice_get(loom_low_op_results(low_op), 0);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
        context, source_op, &plan->ordering));
    return loom_low_lower_bind_value(
        context, loom_view_atomic_cmpxchg_old(source_op), low_old);
  }

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_RMW) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, loom_view_atomic_rmw_result(source_op),
        &result_type));
    if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
      loom_value_id_t operands[] = {low_value, low_resource, low_vaddr,
                                    low_soffset};
      const loom_tied_result_t tied_result = {
          .result_index = 0,
          .operand_index = 0,
      };
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
          context, plan->descriptor_id, operands, IREE_ARRAYSIZE(operands),
          packet_attrs, &result_type, 1, &tied_result, 1, source_op->location,
          &low_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
          context, source_op, &plan->ordering));
      return loom_low_lower_bind_value(
          context, loom_view_atomic_rmw_result(source_op),
          loom_value_slice_get(loom_low_op_results(low_op), 0));
    }

    loom_value_id_t operands[] = {low_vaddr, low_value, low_saddr};
    const iree_host_size_t operand_count =
        plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR ? 3
                                                                           : 2;
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, plan->descriptor_id, operands, operand_count,
        packet_attrs, &result_type, 1, &low_op));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_atomic_post_ordering(
        context, source_op, &plan->ordering));
    return loom_low_lower_bind_value(
        context, loom_view_atomic_rmw_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  if (loom_amdgpu_atomic_uses_buffer_resource(plan)) {
    loom_value_id_t operands[] = {low_value, low_resource, low_vaddr,
                                  low_soffset};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, plan->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), packet_attrs, /*result_types=*/NULL,
        /*result_count=*/0, &low_op));
    return loom_amdgpu_emit_atomic_post_ordering(context, source_op,
                                                 &plan->ordering);
  }

  loom_value_id_t operands[] = {low_vaddr, low_value, low_saddr};
  const iree_host_size_t operand_count =
      plan->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR ? 3
                                                                         : 2;
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_id, operands, operand_count,
      packet_attrs, /*result_types=*/NULL, /*result_count=*/0, &low_op));
  return loom_amdgpu_emit_atomic_post_ordering(context, source_op,
                                               &plan->ordering);
}

iree_status_t loom_amdgpu_low_legality_verify_view_atomic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_view_atomic_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_atomic_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_atomic_select(
          module, loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op, &plan,
          &source_diagnostic, &memory_diagnostic, &diagnostic)) {
    return iree_ok_status();
  }

  iree_string_view_t detail = IREE_SV("source atomic is not representable");
  if (source_diagnostic.rejection_bits != 0) {
    detail = loom_low_source_memory_access_rejection_detail(
        source_diagnostic.rejection_bits);
  } else if (memory_diagnostic.rejection_bits != 0) {
    detail = loom_amdgpu_memory_access_rejection_detail(
        memory_diagnostic.rejection_bits);
  } else {
    detail = loom_amdgpu_atomic_rejection_detail(diagnostic.rejection_bits);
  }
  return loom_target_low_legality_reject(context, provider, op,
                                         IREE_SV("atomic"),
                                         loom_op_name(module, op), detail);
}
