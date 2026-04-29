// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/function_body.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/target/emit/wasm/binary_writer.h"
#include "loom/target/emit/wasm/types.h"

enum {
  LOOM_WASM_OPCODE_END = 0x0B,
  LOOM_WASM_OPCODE_RETURN = 0x0F,
  LOOM_WASM_OPCODE_LOCAL_GET = 0x20,
  LOOM_WASM_OPCODE_LOCAL_SET = 0x21,
  LOOM_WASM_OPCODE_I32_CONST = 0x41,
  LOOM_WASM_OPCODE_I32_ADD = 0x6A,
  LOOM_WASM_OPCODE_I32_SUB = 0x6B,
  LOOM_WASM_OPCODE_I32_MUL = 0x6C,
  LOOM_WASM_OPCODE_I32_LT_U = 0x49,
  LOOM_WASM_OPCODE_F32_ADD = 0x92,
  LOOM_WASM_OPCODE_SIMD_PREFIX = 0xFD,
};

enum {
  LOOM_WASM_SIMD_SUBOPCODE_V128_LOAD = 0x00,
  LOOM_WASM_SIMD_SUBOPCODE_V128_STORE = 0x0B,
  LOOM_WASM_SIMD_SUBOPCODE_V128_CONST = 0x0C,
  LOOM_WASM_SIMD_SUBOPCODE_I8X16_SHUFFLE = 0x0D,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_SPLAT = 0x11,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_EXTRACT_LANE = 0x1B,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_REPLACE_LANE = 0x1C,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_EXTRACT_LANE = 0x1F,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_REPLACE_LANE = 0x20,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_EQ = 0x37,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_NE = 0x38,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_S = 0x39,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_U = 0x3A,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_S = 0x3B,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_U = 0x3C,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_S = 0x3D,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_U = 0x3E,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_S = 0x3F,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_U = 0x40,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_EQ = 0x41,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_LT = 0x43,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_GT = 0x44,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_LE = 0x45,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_GE = 0x46,
  LOOM_WASM_SIMD_SUBOPCODE_V128_BITSELECT = 0x52,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_ADD = 0xAE,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_SUB = 0xB1,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_MUL = 0xB5,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_ADD = 0xE4,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_MUL = 0xE6,
};

enum {
  LOOM_WASM_ENCODING_V128_LOAD =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_V128_LOAD,
  LOOM_WASM_ENCODING_V128_STORE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_V128_STORE,
  LOOM_WASM_ENCODING_V128_CONST =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_V128_CONST,
  LOOM_WASM_ENCODING_I8X16_SHUFFLE = (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
                                     LOOM_WASM_SIMD_SUBOPCODE_I8X16_SHUFFLE,
  LOOM_WASM_ENCODING_I32X4_SPLAT = (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
                                   LOOM_WASM_SIMD_SUBOPCODE_I32X4_SPLAT,
  LOOM_WASM_ENCODING_I32X4_EXTRACT_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_I32X4_EXTRACT_LANE,
  LOOM_WASM_ENCODING_I32X4_REPLACE_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_I32X4_REPLACE_LANE,
  LOOM_WASM_ENCODING_F32X4_EXTRACT_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_F32X4_EXTRACT_LANE,
  LOOM_WASM_ENCODING_F32X4_REPLACE_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_F32X4_REPLACE_LANE,
  LOOM_WASM_ENCODING_I32X4_EQ =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_EQ,
  LOOM_WASM_ENCODING_I32X4_NE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_NE,
  LOOM_WASM_ENCODING_I32X4_LT_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_S,
  LOOM_WASM_ENCODING_I32X4_LT_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_U,
  LOOM_WASM_ENCODING_I32X4_GT_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_S,
  LOOM_WASM_ENCODING_I32X4_GT_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_U,
  LOOM_WASM_ENCODING_I32X4_LE_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_S,
  LOOM_WASM_ENCODING_I32X4_LE_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_U,
  LOOM_WASM_ENCODING_I32X4_GE_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_S,
  LOOM_WASM_ENCODING_I32X4_GE_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_U,
  LOOM_WASM_ENCODING_F32X4_EQ =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_EQ,
  LOOM_WASM_ENCODING_F32X4_LT =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_LT,
  LOOM_WASM_ENCODING_F32X4_GT =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_GT,
  LOOM_WASM_ENCODING_F32X4_LE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_LE,
  LOOM_WASM_ENCODING_F32X4_GE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_GE,
  LOOM_WASM_ENCODING_V128_BITSELECT = (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
                                      LOOM_WASM_SIMD_SUBOPCODE_V128_BITSELECT,
  LOOM_WASM_ENCODING_I32X4_ADD =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_ADD,
  LOOM_WASM_ENCODING_I32X4_SUB =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_SUB,
  LOOM_WASM_ENCODING_I32X4_MUL =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_MUL,
  LOOM_WASM_ENCODING_F32X4_ADD =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_ADD,
  LOOM_WASM_ENCODING_F32X4_MUL =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_MUL,
};

typedef struct loom_wasm_local_entry_t {
  // Descriptor-set-local register class ID from allocation.
  uint16_t descriptor_reg_class_id;
  // Class-local target-id assignment base from allocation.
  uint32_t location_base;
  // Wasm value type stored in this local.
  loom_wasm_value_type_t value_type;
  // Function-local Wasm index assigned by this emitter.
  uint32_t local_index;
} loom_wasm_local_entry_t;

typedef struct loom_wasm_local_layout_t {
  // Allocator used for local metadata arrays.
  iree_allocator_t allocator;
  // Map from class-local allocation target ids to Wasm local indices.
  loom_wasm_local_entry_t* entries;
  // Number of initialized map entries.
  iree_host_size_t entry_count;
  // Allocated map entry capacity.
  iree_host_size_t entry_capacity;
  // Wasm value types indexed by final Wasm local index.
  loom_wasm_value_type_t* local_types;
  // Number of Wasm locals including parameters.
  iree_host_size_t local_type_count;
  // Allocated local type capacity.
  iree_host_size_t local_type_capacity;
  // Number of ABI parameter locals at the start of |local_types|.
  uint32_t parameter_count;
} loom_wasm_local_layout_t;

typedef struct loom_wasm_attr_name_ids_t {
  // Module string ID for wasm.i32.const's immediate payload.
  loom_string_id_t i32_value;
  // Module string ID for wasm.v128.const's low 64-bit immediate payload.
  loom_string_id_t lo64;
  // Module string ID for wasm.v128.const's high 64-bit immediate payload.
  loom_string_id_t hi64;
  // Module string ID for i32x4 lane-immediate payloads.
  loom_string_id_t lane;
  // Module string IDs for i8x16.shuffle byte-lane immediate payloads.
  loom_string_id_t shuffle_lanes[16];
} loom_wasm_attr_name_ids_t;

typedef struct loom_wasm_emit_state_t {
  // Schedule table being emitted.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying class-local target ids.
  const loom_low_allocation_table_t* allocation;
  // Cached module attr-name IDs used to decode low packet immediates.
  loom_wasm_attr_name_ids_t attr_names;
  // Derived Wasm local namespace layout.
  loom_wasm_local_layout_t locals;
  // Mutable body payload writer.
  loom_wasm_binary_writer_t writer;
} loom_wasm_emit_state_t;

static const iree_string_view_t kWasmAttrI32ValueName = IREE_SVL("i32_value");
static const iree_string_view_t kWasmAttrLo64Name = IREE_SVL("lo64");
static const iree_string_view_t kWasmAttrHi64Name = IREE_SVL("hi64");
static const iree_string_view_t kWasmAttrLaneName = IREE_SVL("lane");
static const iree_string_view_t kWasmShuffleLaneAttrNames[16] = {
    IREE_SVL("lane0"),  IREE_SVL("lane1"),  IREE_SVL("lane2"),
    IREE_SVL("lane3"),  IREE_SVL("lane4"),  IREE_SVL("lane5"),
    IREE_SVL("lane6"),  IREE_SVL("lane7"),  IREE_SVL("lane8"),
    IREE_SVL("lane9"),  IREE_SVL("lane10"), IREE_SVL("lane11"),
    IREE_SVL("lane12"), IREE_SVL("lane13"), IREE_SVL("lane14"),
    IREE_SVL("lane15"),
};

static void loom_wasm_attr_name_ids_initialize(
    const loom_module_t* module, loom_wasm_attr_name_ids_t* out_attr_names) {
  *out_attr_names = (loom_wasm_attr_name_ids_t){
      .i32_value = loom_module_lookup_string(module, kWasmAttrI32ValueName),
      .lo64 = loom_module_lookup_string(module, kWasmAttrLo64Name),
      .hi64 = loom_module_lookup_string(module, kWasmAttrHi64Name),
      .lane = loom_module_lookup_string(module, kWasmAttrLaneName),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kWasmShuffleLaneAttrNames);
       ++i) {
    out_attr_names->shuffle_lanes[i] =
        loom_module_lookup_string(module, kWasmShuffleLaneAttrNames[i]);
  }
}

static iree_status_t loom_wasm_write_opcode(loom_wasm_binary_writer_t* writer,
                                            uint32_t encoding_id) {
  if (encoding_id <= UINT8_MAX) {
    return loom_wasm_binary_write_u8(writer, (uint8_t)encoding_id);
  }
  const uint32_t prefix = encoding_id >> 8;
  const uint32_t subopcode = encoding_id & 0xFFu;
  if (prefix != LOOM_WASM_OPCODE_SIMD_PREFIX) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported Wasm opcode prefix 0x%02" PRIx32,
                            prefix);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(writer, (uint8_t)prefix));
  return loom_wasm_binary_write_u32_leb(writer, subopcode);
}

static void loom_wasm_local_layout_deinitialize(
    loom_wasm_local_layout_t* layout) {
  iree_allocator_free(layout->allocator, layout->entries);
  iree_allocator_free(layout->allocator, layout->local_types);
  *layout = (loom_wasm_local_layout_t){0};
}

static iree_status_t loom_wasm_local_layout_reserve_entries(
    loom_wasm_local_layout_t* layout, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= layout->entry_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      layout->entry_capacity ? layout->entry_capacity * 2 : 16;
  if (new_capacity < layout->entry_capacity ||
      new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_mul(new_capacity, sizeof(*layout->entries),
                                  &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local entry table size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(layout->allocator, byte_length,
                                              (void**)&layout->entries));
  layout->entry_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_reserve_types(
    loom_wasm_local_layout_t* layout, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= layout->local_type_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      layout->local_type_capacity ? layout->local_type_capacity * 2 : 16;
  if (new_capacity < layout->local_type_capacity ||
      new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_mul(new_capacity, sizeof(*layout->local_types),
                                  &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local type table size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(layout->allocator, byte_length,
                                              (void**)&layout->local_types));
  layout->local_type_capacity = new_capacity;
  return iree_ok_status();
}

static loom_wasm_local_entry_t* loom_wasm_local_layout_find_entry(
    loom_wasm_local_layout_t* layout, uint16_t descriptor_reg_class_id,
    uint32_t location_base) {
  for (iree_host_size_t i = 0; i < layout->entry_count; ++i) {
    loom_wasm_local_entry_t* entry = &layout->entries[i];
    if (entry->descriptor_reg_class_id == descriptor_reg_class_id &&
        entry->location_base == location_base) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_wasm_local_layout_append_type(
    loom_wasm_local_layout_t* layout, loom_wasm_value_type_t value_type,
    uint32_t* out_local_index) {
  if (layout->local_type_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local index exceeds u32");
  }
  IREE_RETURN_IF_ERROR(loom_wasm_local_layout_reserve_types(
      layout, layout->local_type_count + 1));
  *out_local_index = (uint32_t)layout->local_type_count;
  layout->local_types[layout->local_type_count++] = value_type;
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_add_entry(
    loom_wasm_local_layout_t* layout, uint16_t descriptor_reg_class_id,
    uint32_t location_base, loom_wasm_value_type_t value_type,
    uint32_t local_index) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_local_layout_reserve_entries(layout, layout->entry_count + 1));
  layout->entries[layout->entry_count++] = (loom_wasm_local_entry_t){
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .location_base = location_base,
      .value_type = value_type,
      .local_index = local_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_wasm_validate_target_id_assignment(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    loom_wasm_value_type_t* out_value_type) {
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm value %u is not allocated to a target-local id",
        (unsigned)assignment->value_id);
  }
  if (assignment->location_count != 1 || assignment->unit_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm value %u uses a multi-unit target-id assignment",
        (unsigned)assignment->value_id);
  }
  if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm value %u is not allocated as a register value",
        (unsigned)assignment->value_id);
  }
  return loom_wasm_value_type_from_descriptor_register_class(
      assignment->descriptor_reg_class_id, out_value_type);
}

static iree_status_t loom_wasm_lookup_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id,
    const loom_low_allocation_assignment_t** out_assignment,
    loom_wasm_value_type_t* out_value_type) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_find_assignment(allocation, value_id, NULL);
  if (!assignment) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no allocation",
                            (unsigned)value_id);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_validate_target_id_assignment(
      allocation, assignment, out_value_type));
  *out_assignment = assignment;
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_add_parameter(
    const loom_low_allocation_table_t* allocation,
    loom_wasm_local_layout_t* layout, loom_value_id_t value_id,
    uint32_t parameter_index) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  loom_wasm_value_type_t value_type = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_lookup_assignment(allocation, value_id,
                                                   &assignment, &value_type));
  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_local_layout_append_type(layout, value_type, &local_index));
  if (local_index != parameter_index) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm parameter index overflow");
  }

  loom_wasm_local_entry_t* existing = loom_wasm_local_layout_find_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base);
  if (existing) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm parameters %u and %" PRIu32 " share allocator target id %" PRIu32
        " for one register class",
        existing->local_index, parameter_index, assignment->location_base);
  }
  return loom_wasm_local_layout_add_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base,
      value_type, parameter_index);
}

static iree_status_t loom_wasm_local_layout_add_assignment(
    const loom_low_allocation_table_t* allocation,
    loom_wasm_local_layout_t* layout,
    const loom_low_allocation_assignment_t* assignment) {
  loom_wasm_value_type_t value_type = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_validate_target_id_assignment(
      allocation, assignment, &value_type));
  loom_wasm_local_entry_t* existing = loom_wasm_local_layout_find_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base);
  if (existing) {
    if (existing->value_type != value_type) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Wasm target id %" PRIu32 " changes value type",
                              assignment->location_base);
    }
    return iree_ok_status();
  }

  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_local_layout_append_type(layout, value_type, &local_index));
  return loom_wasm_local_layout_add_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base,
      value_type, local_index);
}

static iree_status_t loom_wasm_build_local_layout(
    const loom_low_allocation_table_t* allocation, iree_allocator_t allocator,
    loom_wasm_local_layout_t* out_layout) {
  *out_layout = (loom_wasm_local_layout_t){
      .allocator = allocator,
  };

  loom_func_like_t function = loom_func_like_cast(
      allocation->module, (loom_op_t*)allocation->function_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm emission requires a func-like low function");
  }

  uint16_t parameter_count = 0;
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function, &parameter_count);
  out_layout->parameter_count = parameter_count;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < parameter_count && iree_status_is_ok(status); ++i) {
    status = loom_wasm_local_layout_add_parameter(allocation, out_layout,
                                                  parameter_ids[i], i);
  }
  for (iree_host_size_t i = 0;
       i < allocation->assignment_count && iree_status_is_ok(status); ++i) {
    status = loom_wasm_local_layout_add_assignment(allocation, out_layout,
                                                   &allocation->assignments[i]);
  }
  if (!iree_status_is_ok(status)) {
    loom_wasm_local_layout_deinitialize(out_layout);
  }
  return status;
}

static iree_status_t loom_wasm_lookup_local(
    loom_wasm_emit_state_t* state, loom_value_id_t value_id,
    uint32_t* out_local_index, loom_wasm_value_type_t* out_value_type) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  loom_wasm_value_type_t value_type = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_lookup_assignment(state->allocation, value_id,
                                                   &assignment, &value_type));
  loom_wasm_local_entry_t* entry = loom_wasm_local_layout_find_entry(
      &state->locals, assignment->descriptor_reg_class_id,
      assignment->location_base);
  if (!entry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no local mapping",
                            (unsigned)value_id);
  }
  *out_local_index = entry->local_index;
  if (out_value_type) {
    *out_value_type = entry->value_type;
  }
  return iree_ok_status();
}

static const loom_named_attr_t* loom_wasm_find_named_attr_by_id(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_wasm_read_i64_attr(loom_named_attr_slice_t attrs,
                                             iree_string_view_t name,
                                             loom_string_id_t name_id,
                                             int64_t* out_value) {
  const loom_named_attr_t* attr =
      loom_wasm_find_named_attr_by_id(attrs, name_id);
  if (!attr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm emission missing required '%.*s' attribute",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm attribute '%.*s' must be i64", (int)name.size,
                            name.data);
  }
  *out_value = attr->value.i64;
  return iree_ok_status();
}

static iree_status_t loom_wasm_read_u8_attr(loom_named_attr_slice_t attrs,
                                            iree_string_view_t name,
                                            loom_string_id_t name_id,
                                            uint8_t maximum_value,
                                            uint8_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_read_i64_attr(attrs, name, name_id, &value));
  if (value < 0 || value > maximum_value) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm attribute '%.*s' is outside [0, %" PRIu8 "]",
                            (int)name.size, name.data, maximum_value);
  }
  *out_value = (uint8_t)value;
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_local_get(loom_wasm_emit_state_t* state,
                                              loom_value_id_t value_id) {
  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_lookup_local(state, value_id, &local_index, NULL));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOCAL_GET));
  return loom_wasm_binary_write_u32_leb(&state->writer, local_index);
}

static iree_status_t loom_wasm_emit_local_set(loom_wasm_emit_state_t* state,
                                              loom_value_id_t value_id) {
  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_lookup_local(state, value_id, &local_index, NULL));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOCAL_SET));
  return loom_wasm_binary_write_u32_leb(&state->writer, local_index);
}

static iree_status_t loom_wasm_emit_memarg(loom_wasm_emit_state_t* state,
                                           uint32_t alignment_exponent,
                                           uint32_t offset) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(&state->writer, alignment_exponent));
  return loom_wasm_binary_write_u32_leb(&state->writer, offset);
}

static iree_status_t loom_wasm_emit_i32_const(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_const_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wasm.i32.const must be a unary low.const");
  }
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_read_i64_attr(loom_low_const_attrs(op), kWasmAttrI32ValueName,
                              state->attr_names.i32_value, &value));
  if (value < INT32_MIN || value > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "wasm.i32.const value is outside i32 range");
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_i32_leb(&state->writer, (int32_t)value));
  return loom_wasm_emit_local_set(state, loom_low_const_result(op));
}

static iree_status_t loom_wasm_emit_v128_const(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_const_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wasm.v128.const must be a unary low.const");
  }
  int64_t low_bits = 0;
  int64_t high_bits = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_read_i64_attr(loom_low_const_attrs(op), kWasmAttrLo64Name,
                              state->attr_names.lo64, &low_bits));
  IREE_RETURN_IF_ERROR(
      loom_wasm_read_i64_attr(loom_low_const_attrs(op), kWasmAttrHi64Name,
                              state->attr_names.hi64, &high_bits));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u64_le(&state->writer, (uint64_t)low_bits));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u64_le(&state->writer, (uint64_t)high_bits));
  return loom_wasm_emit_local_set(state, loom_low_const_result(op));
}

static iree_status_t loom_wasm_emit_unary_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 1 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm unary packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_binary_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm binary packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_ternary_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 3 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm ternary packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[2]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_lane_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  const bool extracts_lane =
      descriptor->encoding_id == LOOM_WASM_ENCODING_I32X4_EXTRACT_LANE ||
      descriptor->encoding_id == LOOM_WASM_ENCODING_F32X4_EXTRACT_LANE;
  const uint16_t expected_operand_count = extracts_lane ? 1 : 2;
  if (!loom_low_op_isa(op) || op->operand_count != expected_operand_count ||
      op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm lane packet shape is invalid");
  }
  loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  uint8_t lane = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_read_u8_attr(attrs, kWasmAttrLaneName,
                                              state->attr_names.lane,
                                              /*maximum_value=*/3, &lane));
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  for (iree_host_size_t i = 0; i < operands.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, lane));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_i8x16_shuffle(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wasm.i8x16.shuffle packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kWasmShuffleLaneAttrNames);
       ++i) {
    uint8_t lane = 0;
    IREE_RETURN_IF_ERROR(loom_wasm_read_u8_attr(
        attrs, kWasmShuffleLaneAttrNames[i], state->attr_names.shuffle_lanes[i],
        /*maximum_value=*/31, &lane));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, lane));
  }
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_v128_load(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 1 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wasm.v128.load packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memarg(state, /*alignment_exponent=*/4,
                                             /*offset=*/0));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_v128_store(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wasm.v128.store packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_memarg(state, /*alignment_exponent=*/4, /*offset=*/0);
}

static iree_status_t loom_wasm_emit_descriptor_packet(
    loom_wasm_emit_state_t* state, const loom_low_packet_view_t* packet) {
  switch (packet->descriptor->encoding_id) {
    case LOOM_WASM_OPCODE_I32_CONST:
      return loom_wasm_emit_i32_const(state, packet->node->op,
                                      packet->descriptor);
    case LOOM_WASM_ENCODING_V128_CONST:
      return loom_wasm_emit_v128_const(state, packet->node->op,
                                       packet->descriptor);
    case LOOM_WASM_OPCODE_I32_ADD:
    case LOOM_WASM_OPCODE_I32_SUB:
    case LOOM_WASM_OPCODE_I32_MUL:
    case LOOM_WASM_OPCODE_I32_LT_U:
    case LOOM_WASM_OPCODE_F32_ADD:
    case LOOM_WASM_ENCODING_I32X4_EQ:
    case LOOM_WASM_ENCODING_I32X4_NE:
    case LOOM_WASM_ENCODING_I32X4_LT_S:
    case LOOM_WASM_ENCODING_I32X4_LT_U:
    case LOOM_WASM_ENCODING_I32X4_GT_S:
    case LOOM_WASM_ENCODING_I32X4_GT_U:
    case LOOM_WASM_ENCODING_I32X4_LE_S:
    case LOOM_WASM_ENCODING_I32X4_LE_U:
    case LOOM_WASM_ENCODING_I32X4_GE_S:
    case LOOM_WASM_ENCODING_I32X4_GE_U:
    case LOOM_WASM_ENCODING_I32X4_ADD:
    case LOOM_WASM_ENCODING_I32X4_SUB:
    case LOOM_WASM_ENCODING_I32X4_MUL:
    case LOOM_WASM_ENCODING_F32X4_EQ:
    case LOOM_WASM_ENCODING_F32X4_LT:
    case LOOM_WASM_ENCODING_F32X4_GT:
    case LOOM_WASM_ENCODING_F32X4_LE:
    case LOOM_WASM_ENCODING_F32X4_GE:
    case LOOM_WASM_ENCODING_F32X4_ADD:
    case LOOM_WASM_ENCODING_F32X4_MUL:
      return loom_wasm_emit_binary_stack_op(state, packet->node->op,
                                            packet->descriptor);
    case LOOM_WASM_ENCODING_V128_BITSELECT:
      return loom_wasm_emit_ternary_stack_op(state, packet->node->op,
                                             packet->descriptor);
    case LOOM_WASM_ENCODING_I8X16_SHUFFLE:
      return loom_wasm_emit_i8x16_shuffle(state, packet->node->op,
                                          packet->descriptor);
    case LOOM_WASM_ENCODING_I32X4_SPLAT:
      return loom_wasm_emit_unary_stack_op(state, packet->node->op,
                                           packet->descriptor);
    case LOOM_WASM_ENCODING_I32X4_EXTRACT_LANE:
    case LOOM_WASM_ENCODING_F32X4_EXTRACT_LANE:
    case LOOM_WASM_ENCODING_I32X4_REPLACE_LANE:
    case LOOM_WASM_ENCODING_F32X4_REPLACE_LANE:
      return loom_wasm_emit_lane_stack_op(state, packet->node->op,
                                          packet->descriptor);
    case LOOM_WASM_ENCODING_V128_LOAD:
      return loom_wasm_emit_v128_load(state, packet->node->op,
                                      packet->descriptor);
    case LOOM_WASM_ENCODING_V128_STORE:
      return loom_wasm_emit_v128_store(state, packet->node->op,
                                       packet->descriptor);
    default: {
      iree_string_view_t key = IREE_SV("<unknown>");
      (void)loom_low_descriptor_set_string(
          state->schedule->target.descriptor_set,
          packet->descriptor->key_string_offset, &key);
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Wasm descriptor '%.*s' is unsupported",
                              (int)key.size, key.data);
    }
  }
}

static iree_status_t loom_wasm_emit_low_copy(loom_wasm_emit_state_t* state,
                                             const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_local_get(state, loom_low_copy_source(op)));
  return loom_wasm_emit_local_set(state, loom_low_copy_result(op));
}

static iree_status_t loom_wasm_emit_low_return(loom_wasm_emit_state_t* state,
                                               const loom_op_t* op) {
  loom_value_slice_t values = loom_low_return_values(op);
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, values.values[i]));
  }
  return loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_RETURN);
}

static iree_status_t loom_wasm_emit_structural_packet(
    loom_wasm_emit_state_t* state, const loom_op_t* op) {
  if (loom_low_copy_isa(op)) {
    return loom_wasm_emit_low_copy(state, op);
  }
  if (loom_low_return_isa(op)) {
    return loom_wasm_emit_low_return(state, op);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "unsupported structural low op in Wasm function-body emission");
}

static iree_status_t loom_wasm_emit_packet(
    loom_wasm_emit_state_t* state, const loom_low_packet_view_t* packet) {
  if (packet->descriptor) {
    return loom_wasm_emit_descriptor_packet(state, packet);
  }
  return loom_wasm_emit_structural_packet(state, packet->node->op);
}

static iree_status_t loom_wasm_emit_local_declarations(
    loom_wasm_emit_state_t* state) {
  uint32_t declaration_count = 0;
  for (iree_host_size_t i = state->locals.parameter_count;
       i < state->locals.local_type_count;) {
    ++declaration_count;
    loom_wasm_value_type_t value_type = state->locals.local_types[i++];
    while (i < state->locals.local_type_count &&
           state->locals.local_types[i] == value_type) {
      ++i;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(&state->writer, declaration_count));
  for (iree_host_size_t i = state->locals.parameter_count;
       i < state->locals.local_type_count;) {
    loom_wasm_value_type_t value_type = state->locals.local_types[i++];
    uint32_t run_length = 1;
    while (i < state->locals.local_type_count &&
           state->locals.local_types[i] == value_type) {
      ++run_length;
      ++i;
    }
    IREE_RETURN_IF_ERROR(
        loom_wasm_binary_write_u32_leb(&state->writer, run_length));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, value_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_function_body_payload(
    loom_wasm_emit_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_declarations(state));
  const loom_low_schedule_block_t* block = &state->schedule->blocks[0];
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    iree_host_size_t packet_index =
        (iree_host_size_t)block->scheduled_node_start + i;
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        state->schedule, state->allocation, packet_index, &packet));
    IREE_RETURN_IF_ERROR(loom_wasm_emit_packet(state, &packet));
  }
  return loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_END);
}

static iree_status_t loom_wasm_validate_tables(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation) {
  if (!schedule || !allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule and allocation tables are required");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  if (schedule->target.descriptor_set !=
      loom_wasm_core_simd128_descriptor_set()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm emission requires wasm.core.simd128");
  }
  if (schedule->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm function-body emission currently requires one low block");
  }
  if (allocation->spill_count != 0 || allocation->spill_plan_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm emission requires unspilled allocation tables");
  }
  return iree_ok_status();
}

void loom_wasm_function_body_deinitialize(loom_wasm_function_body_t* body,
                                          iree_allocator_t allocator) {
  if (!body) {
    return;
  }
  iree_allocator_free(allocator, body->data);
  *body = (loom_wasm_function_body_t){0};
}

iree_status_t loom_wasm_emit_function_body(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation, iree_allocator_t allocator,
    loom_wasm_function_body_t* out_body) {
  if (!out_body) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm function-body output is required");
  }
  *out_body = (loom_wasm_function_body_t){0};
  IREE_RETURN_IF_ERROR(loom_wasm_validate_tables(schedule, allocation));

  loom_wasm_emit_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
  };
  loom_wasm_attr_name_ids_initialize(schedule->module, &state.attr_names);
  loom_wasm_binary_writer_initialize(allocator, &state.writer);
  iree_status_t status =
      loom_wasm_build_local_layout(allocation, allocator, &state.locals);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_emit_function_body_payload(&state);
  }

  loom_wasm_binary_writer_t output_writer;
  loom_wasm_binary_writer_initialize(allocator, &output_writer);
  if (iree_status_is_ok(status) && state.writer.length > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm function body exceeds u32 size");
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_u32_leb(&output_writer,
                                            (uint32_t)state.writer.length);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_bytes(&output_writer, state.writer.data,
                                          state.writer.length);
  }
  if (iree_status_is_ok(status) && state.locals.local_type_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm local count exceeds u32");
  }
  if (iree_status_is_ok(status)) {
    *out_body = (loom_wasm_function_body_t){
        .data = output_writer.data,
        .data_length = output_writer.length,
        .body_length = state.writer.length,
        .parameter_count = state.locals.parameter_count,
        .local_count = (uint32_t)state.locals.local_type_count,
    };
    output_writer.data = NULL;
  }

  loom_wasm_binary_writer_deinitialize(&output_writer);
  loom_wasm_binary_writer_deinitialize(&state.writer);
  loom_wasm_local_layout_deinitialize(&state.locals);
  return status;
}
