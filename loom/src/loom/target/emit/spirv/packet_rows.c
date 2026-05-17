// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/packet_rows.h"

#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/emit/spirv/binary_format.h"

static const loom_spirv_packet_row_t kSpirvLogicalCorePacketRows[] = {
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_I32] =
        {
            .opcode = LOOM_SPIRV_OP_CONSTANT,
            .form = LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT,
            .result_kind = LOOM_SPIRV_VALUE_I32,
            .result_count = 1,
            .immediate_index = 0,
            .literal_word_count = 1,
        },
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_OFFSET64] =
        {
            .opcode = LOOM_SPIRV_OP_CONSTANT,
            .form = LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT,
            .result_kind = LOOM_SPIRV_VALUE_U64,
            .result_count = 1,
            .immediate_index = 0,
            .literal_word_count = 2,
        },
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IADD_I32] =
        {
            .opcode = LOOM_SPIRV_OP_I_ADD,
            .form = LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE,
            .result_kind = LOOM_SPIRV_VALUE_I32,
            .operand_kinds = {LOOM_SPIRV_VALUE_I32, LOOM_SPIRV_VALUE_I32},
            .result_count = 1,
            .operand_count = 2,
            .immediate_index = LOOM_SPIRV_PACKET_IMMEDIATE_NONE,
        },
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IADD_OFFSET64] =
        {
            .opcode = LOOM_SPIRV_OP_I_ADD,
            .form = LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE,
            .result_kind = LOOM_SPIRV_VALUE_U64,
            .operand_kinds = {LOOM_SPIRV_VALUE_U64, LOOM_SPIRV_VALUE_U64},
            .result_count = 1,
            .operand_count = 2,
            .immediate_index = LOOM_SPIRV_PACKET_IMMEDIATE_NONE,
        },
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_PTR_ACCESS_CHAIN_STORAGE_BUFFER_BYTE_OFFSET] =
        {
            .opcode = LOOM_SPIRV_OP_PTR_ACCESS_CHAIN,
            .form = LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN,
            .result_kind = LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32,
            .operand_kinds =
                {
                    LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32,
                    LOOM_SPIRV_VALUE_U64,
                },
            .result_count = 1,
            .operand_count = 2,
            .immediate_index = LOOM_SPIRV_PACKET_IMMEDIATE_NONE,
        },
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_STORAGE_BUFFER_I32] =
        {
            .opcode = LOOM_SPIRV_OP_LOAD,
            .form = LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED,
            .result_kind = LOOM_SPIRV_VALUE_I32,
            .operand_kinds =
                {
                    LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32,
                },
            .result_count = 1,
            .operand_count = 1,
            .immediate_index = LOOM_SPIRV_PACKET_IMMEDIATE_NONE,
            .memory_alignment = 4,
        },
    [SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_STORE_STORAGE_BUFFER_I32] =
        {
            .opcode = LOOM_SPIRV_OP_STORE,
            .form = LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED,
            .operand_kinds =
                {
                    LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32,
                    LOOM_SPIRV_VALUE_I32,
                },
            .operand_count = 2,
            .immediate_index = LOOM_SPIRV_PACKET_IMMEDIATE_NONE,
            .memory_alignment = 4,
        },
};

const loom_spirv_packet_row_t* loom_spirv_packet_row_for_descriptor_ordinal(
    uint32_t descriptor_ordinal) {
  if (descriptor_ordinal >= IREE_ARRAYSIZE(kSpirvLogicalCorePacketRows)) {
    return NULL;
  }
  const loom_spirv_packet_row_t* row =
      &kSpirvLogicalCorePacketRows[descriptor_ordinal];
  return row->form == LOOM_SPIRV_PACKET_FORM_UNSUPPORTED ? NULL : row;
}
