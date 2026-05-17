// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-owned SPIR-V packet emission rows.
//
// These rows describe how selected target-low descriptors serialize to SPIR-V
// instructions. They are keyed by dense descriptor ordinal so the binary
// emitter can dispatch without string comparisons or descriptor-key switches.

#ifndef LOOM_TARGET_EMIT_SPIRV_PACKET_ROWS_H_
#define LOOM_TARGET_EMIT_SPIRV_PACKET_ROWS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SPIRV_PACKET_IMMEDIATE_NONE UINT8_MAX

typedef enum loom_spirv_value_kind_e {
  LOOM_SPIRV_VALUE_UNKNOWN = 0,
  LOOM_SPIRV_VALUE_I32 = 1,
  LOOM_SPIRV_VALUE_U64 = 2,
  LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32 = 3,
} loom_spirv_value_kind_t;

typedef enum loom_spirv_packet_form_e {
  LOOM_SPIRV_PACKET_FORM_UNSUPPORTED = 0,
  LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT = 1,
  LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE = 2,
  LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN = 3,
  LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED = 4,
  LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED = 5,
} loom_spirv_packet_form_t;

typedef struct loom_spirv_packet_row_t {
  // SPIR-V instruction opcode.
  uint32_t opcode;
  // Emission algorithm selected for this descriptor.
  loom_spirv_packet_form_t form;
  // Result type/value category, or UNKNOWN for result-less packets.
  loom_spirv_value_kind_t result_kind;
  // Required value category for each packet operand.
  loom_spirv_value_kind_t operand_kinds[2];
  // Expected packet result count.
  uint8_t result_count;
  // Expected packet operand count.
  uint8_t operand_count;
  // Descriptor-local immediate index read by the row.
  uint8_t immediate_index;
  // Number of literal words emitted for INTEGER_CONSTANT rows.
  uint8_t literal_word_count;
  // Alignment operand for aligned memory access rows.
  uint8_t memory_alignment;
} loom_spirv_packet_row_t;

// Returns the packet row for |descriptor_ordinal|, or NULL when the descriptor
// has no binary emission row yet.
const loom_spirv_packet_row_t* loom_spirv_packet_row_for_descriptor_ordinal(
    uint32_t descriptor_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_PACKET_ROWS_H_
